import streamlit as st
import tempfile

from rag_utils import (
    build_rag,
    rerank_documents,
)

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
            llm = rag["llm"]
            prompt = rag["prompt"]

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