import streamlit as st
import tempfile

from rag_utils import (
    build_rag,
    rerank_documents,
    balanced_multi_document_retrieval,
)

def is_multi_document_question(question, uploaded_names):
    """
    Detect questions that explicitly require information
    from multiple uploaded documents.
    """

    question_lower = question.lower()

    # Explicit multi-document intent
    multi_document_phrases = [
        "compare",
        "comparison",
        "compare the",
        "difference between",
        "differences between",
        "versus",
        " vs ",
        "both documents",
        "both pdfs",
        "both files",
        "each document",
        "each pdf",
        "each file",
        "all documents",
        "all pdfs",
        "all files",
        "uploaded documents",
        "uploaded pdfs",
        "uploaded files",
        "in both",
        "from both",
    ]

    if any(phrase in question_lower for phrase in multi_document_phrases):
        return True

    # Also detect when the user explicitly mentions
    # two or more uploaded document names.
    mentioned_documents = 0

    for filename in uploaded_names:
        filename_without_ext = filename.rsplit(".", 1)[0].lower()

        # Use words from the filename rather than requiring
        # the complete filename.
        filename_words = [
            word for word in filename_without_ext.replace("_", " ").split()
            if len(word) >= 4
        ]

        if any(word in question_lower for word in filename_words):
            mentioned_documents += 1

    return mentioned_documents >= 2

st.set_page_config(page_title="Hybrid RAG")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

st.title("Hybrid RAG")

with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

if uploaded_files:

    pdf_paths = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_paths.append((tmp.name, uploaded_file.name))

    uploaded_names = [name for _, name in pdf_paths]

    if (
        "qa_chain" not in st.session_state
        or "document_retrievers" not in st.session_state.qa_chain
        or st.session_state.current_pdf != uploaded_names
    ):

        with st.spinner("Processing PDF..."):
            try:
                st.session_state.qa_chain = build_rag(pdf_paths)

            except Exception as e:
                st.error(str(e))
                st.stop()
        st.session_state.current_pdf = uploaded_names
        st.session_state.chat_history = []

        st.success(f"{len(uploaded_files)} documents processed successfully.")

    question = st.chat_input("Ask a question...")

    if question:

        with st.spinner("Searching the document and generating answer..."):

            rag = st.session_state.qa_chain

            retriever = rag["retriever"]
            document_retrievers = rag["document_retrievers"]
            llm = rag["llm"]
            prompt = rag["prompt"]

            is_multi_document = (
                len(uploaded_names) > 1
                and is_multi_document_question(
                    question,
                    uploaded_names
                )
            )

            if is_multi_document:
                reranked_docs = balanced_multi_document_retrieval(
                    question,
                    document_retrievers
                )
            else:
                docs = retriever.invoke(question)

                reranked_docs = rerank_documents(
                    question,
                    docs
                )

            context = "\n\n".join(
                doc.page_content for doc in reranked_docs
            )

            sources = []

            for doc in reranked_docs:
                source_file = doc.metadata.get("source_file", "Unknown document")
                page = doc.metadata.get("page")
                chunk_id = doc.metadata.get("chunk_id", "N/A")
                score = doc.metadata.get("rerank_score")

                if page is not None:
                    source_text = f"{source_file} — Page {page + 1}"
                else:
                    source_text = source_file

                sources.append({
                    "source": source_text,
                    "chunk_id": chunk_id,
                    "score": score
                })

            formatted_prompt = prompt.format(
                context=context,
                question=question
            )

            result = llm.invoke(formatted_prompt)

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": result.content,
                "sources": sources
            }
        )

        st.rerun()

    if st.session_state.chat_history:

        st.divider()
        st.subheader("Conversation")

        for chat in st.session_state.chat_history:

            with st.container(border=True):

                st.markdown(
                    f":material/account_circle: **You**\n\n{chat['question']}"
                )
                st.markdown("---")
                st.markdown(
                    ":material/smart_toy: **Chatbot**"
                )
                st.markdown(chat["answer"])

                if chat.get("sources"):
                    with st.expander("📚 Sources & Retrieval Details"):

                        for item in chat["sources"]:
                            score = item.get("score")

                            if score is not None:
                                score_text = f"{score:.4f}"
                            else:
                                score_text = "N/A"

                            st.markdown(
                                f"- **{item['source']}**  \n"
                                f"  Chunk ID: `{item['chunk_id']}`  |  "
                                f"Cross-Encoder Score: `{score_text}`"
                            )

            st.divider()


    if st.button(":material/delete: Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()