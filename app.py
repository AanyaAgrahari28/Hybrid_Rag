import streamlit as st
import tempfile
import time
from collections import defaultdict

from rag_utils import (
    build_rag,
    rerank_documents,
    balanced_multi_document_retrieval,
    filter_documents_by_metadata,
    prioritize_recent_documents,
    rewrite_query,
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
    top_k = st.slider(
        "Number of relevant chunks",
        min_value=2,
        max_value=10,
        value=6,
        step=1
    )    

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )

    uploaded_names = [file.name for file in uploaded_files]

    selected_document = st.selectbox(
        "Filter by document",
        ["All documents"] + uploaded_names
    )

if uploaded_files:

    document_paths = []

    for uploaded_file in uploaded_files:

        extension = uploaded_file.name.rsplit(".", 1)[-1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{extension}"
        ) as tmp:

            tmp.write(uploaded_file.read())

            document_paths.append(
                (tmp.name, uploaded_file.name)
            )

    uploaded_names = [
        name for _, name in document_paths
    ]

    if (
        "qa_chain" not in st.session_state
        or "document_retrievers" not in st.session_state.qa_chain
        or st.session_state.current_pdf != uploaded_names
    ):

        with st.spinner("Processing documents..."):
            try:
                st.session_state.qa_chain = build_rag(document_paths)

            except Exception as e:
                st.error(str(e))
                st.stop()
        st.session_state.current_pdf = uploaded_names
        st.session_state.chat_history = []

        st.success(f"{len(uploaded_files)} documents processed successfully.")

    question = st.chat_input("Ask a question...")

    if question:

        with st.spinner("Searching the document and generating answer..."):

            retrieval_start = time.time()

            rag = st.session_state.qa_chain

            retriever = rag["retriever"]
            document_retrievers = rag["document_retrievers"]
            llm = rag["llm"]
            prompt = rag["prompt"]

            rewritten_question = rewrite_query(question, llm)
            initial_retrieved_count = 0

            is_multi_document = (
                len(uploaded_names) > 1
                and is_multi_document_question(
                    question,
                    uploaded_names
                )
            )

            if is_multi_document:
                reranked_docs = balanced_multi_document_retrieval(
                    rewritten_question,
                    document_retrievers,
                    total_k=top_k
                )
            else:
                docs = retriever.invoke(rewritten_question)
                initial_retrieved_count = len(docs)
                if selected_document != "All documents":
                    docs = filter_documents_by_metadata(
                        docs,
                        source_file=selected_document
                    )

                reranked_docs = rerank_documents(
                    rewritten_question,
                    docs,
                    top_k=top_k
                )
                reranked_docs = prioritize_recent_documents(reranked_docs)
                reranked_docs = reranked_docs[:top_k]

            retrieval_time = time.time() - retrieval_start

            context_parts = []

            for i, doc in enumerate(reranked_docs, start=1):

                doc.metadata["citation_id"] = i

                source_file = doc.metadata.get(
                    "source_file",
                    "Unknown"
                )

                page = doc.metadata.get(
                    "page",
                    "N/A"
                )

                chunk_id = doc.metadata.get(
                    "chunk_id",
                    "N/A"
                )

                context_parts.append(
                    f"[SOURCE {i}]\n"
                    f"Document: {source_file}\n"
                    f"Page: {page}\n"
                    f"Chunk: {chunk_id}\n"
                    f"Content:\n"
                    f"{doc.page_content}"
                )

            context = "\n\n".join(context_parts)

            sources = []

            for i, doc in enumerate(reranked_docs, start=1):

                sources.append({
                    "citation_id": i,
                    "source": doc.metadata.get(
                        "source_file",
                        "Unknown"
                    ),
                    "page": doc.metadata.get(
                        "page",
                        "N/A"
                    ),
                    "chunk_id": doc.metadata.get(
                        "chunk_id",
                        "N/A"
                    ),
                    "score": doc.metadata.get(
                        "rerank_score",
                        0
                    ),
                })

            formatted_prompt = prompt.format(
                context=context,
                question=question
            )

            result = llm.invoke(formatted_prompt)

            st.write("### 🔍 Retrieval Diagnostics")

            st.write(f"**Retrieval latency:** {retrieval_time:.3f} seconds")
            st.write(f"**Final chunks used:** {len(reranked_docs)}")

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": result.content,
                "sources": sources,
                "retrieval_time": retrieval_time,
                "final_chunks": len(reranked_docs)
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

                with st.expander("📚 Sources & Retrieval Details"):

                    st.write("### 🔍 Retrieval Diagnostics:")
                    st.write(
                        f"**Retrieval latency:** "
                        f"{chat['retrieval_time']:.3f} seconds"
                    )
                    st.write(
                        f"**Final chunks used:** "
                        f"{chat['final_chunks']}"
                    )

                    st.write("### 📄 Sources")

                    grouped_sources = defaultdict(list)

                    for item in chat["sources"]:
                        grouped_sources[item["source"]].append(item)

                    for source_file, items in grouped_sources.items():

                        st.markdown(f"**📄 {source_file}**")

                        for item in items:
                            st.write(
                                f"**[{item['citation_id']}]** "
                                f"Page {item['page']} · "
                                f"Chunk {item['chunk_id']} · "
                                f"Rerank score: "
                                f"{item['score']:.2f}"
                            )

            st.divider()


    if st.button(":material/delete: Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()