import streamlit as st
import tempfile
import time
import os
import shutil
from collections import defaultdict

from rag_utils import (
    build_rag,
    rerank_documents,
    balanced_multi_document_retrieval,
    filter_documents_by_metadata,
    prioritize_recent_documents,
    rewrite_query,
    load_existing_rag,
    get_knowledge_base_documents,
    retrieve_with_query_fusion,
    route_documents,
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

def get_authorized_documents(knowledge_base_documents, allowed_departments):
    """Return only documents the current role is authorized to access."""

    if "ALL" in allowed_departments:
        return knowledge_base_documents

    return [
        (file_path, file_name)
        for file_path, file_name in knowledge_base_documents
        if os.path.basename(
            os.path.dirname(file_path)
        ) in allowed_departments
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
    
uploaded_names = []

# Role-based department access
ROLE_DEPARTMENTS = {
    "Admin": ["ALL"],
    "HR User": ["People_HR"],
    "IT User": ["Engineering_IT", "Information_Security"],
    "Finance User": ["Finance"],
    "Legal/Compliance User": ["Legal_Compliance"],
}

st.title("Hybrid RAG")

with st.sidebar:
    st.header("📄 Documents")

    role = st.selectbox(
        "Role",
        list(ROLE_DEPARTMENTS.keys())
    )
    # Reset the RAG session when the user's role changes
    if "active_role" not in st.session_state:
        st.session_state.active_role = role

    elif st.session_state.active_role != role:
        st.session_state.active_role = role
        st.session_state.qa_chain = None
        st.session_state.chat_history = []
        st.rerun()

    allowed_departments = ROLE_DEPARTMENTS[role]

    knowledge_base_documents = get_knowledge_base_documents()

    authorized_documents = get_authorized_documents(
    knowledge_base_documents,
    allowed_departments
    )

    top_k = st.slider(
        "Number of relevant chunks",
        min_value=2,
        max_value=10,
        value=6,
        step=1
    )

    if role == "Admin":
        st.markdown("### 📚 Knowledge Base")

        departments = defaultdict(list)

        for file_path, file_name in knowledge_base_documents:
            department = os.path.basename(
                os.path.dirname(file_path)
            )
            departments[department].append(file_name)

        for department, files in sorted(departments.items()):
            with st.expander(f"📁 {department}"):
                for file_name in sorted(files):
                    st.write(f"📄 {file_name}")

        st.markdown("### ⬆️ Upload Documents")

        upload_department = st.selectbox(
            "Department",
            [
                "Customer_Support",
                "Engineering_IT",
                "Finance",
                "Information_Security",
                "Legal_Compliance",
                "Operations",
                "People_HR"
            ]
        )

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt", "md", "xml"],
            accept_multiple_files=True
        )


    else:
        uploaded_files = []
        uploaded_names = []

    all_document_names = [
        name for _, name in authorized_documents
    ]

    # Include documents uploaded in the current Admin session
    if role == "Admin" and uploaded_files:
        all_document_names = list(
            dict.fromkeys(
                all_document_names
                + [file.name for file in uploaded_files]
            )
        )

    selected_document = st.selectbox(
        "Filter by document",
        ["All documents"] + all_document_names
    )

# Load permanent knowledge base on startup
if (
    "qa_chain" not in st.session_state
    or st.session_state.qa_chain is None
):
    knowledge_base_documents = authorized_documents

    if knowledge_base_documents:
        with st.spinner("Loading enterprise knowledge base..."):
            try:
                st.session_state.qa_chain = load_existing_rag(
                    authorized_documents
                )
                st.session_state.current_pdf = [
                    name for _, name in knowledge_base_documents
                ]
                st.session_state.chat_history = []
            except Exception as e:
                st.error(str(e))
                st.stop()

if "qa_chain" in st.session_state:

    if uploaded_files:

        document_paths = []

        department_path = os.path.join(
            "knowledge_base",
            "documents",
            upload_department
        )

        os.makedirs(
            department_path,
            exist_ok=True
        )

        for uploaded_file in uploaded_files:

            permanent_path = os.path.join(
                department_path,
                uploaded_file.name
            )

            with open(permanent_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            document_paths.append(
                (
                    permanent_path,
                    uploaded_file.name
                )
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
                    st.session_state.qa_chain = load_existing_rag(
                        get_authorized_documents(
                            get_knowledge_base_documents(),
                            allowed_departments
                        )
                    )

                except Exception as e:
                    st.error(str(e))
                    st.stop()

            st.session_state.current_pdf = uploaded_names
            st.session_state.chat_history = []

            st.success(
                f"{len(uploaded_files)} documents processed successfully."
            )

            st.rerun()

    question = st.chat_input("Ask a question...")

    if question:

        with st.spinner("Searching the document and generating answer..."):

            retrieval_start = time.time()

            rag = st.session_state.qa_chain

            retriever = rag["retriever"]
            document_retrievers = rag["document_retrievers"]
            document_router = rag["document_router"]
            llm = rag["llm"]
            prompt = rag["prompt"]

            # Measure query rewriting
            rewrite_start = time.time()
            rewritten_question = rewrite_query(question, llm)
            rewrite_time = time.time() - rewrite_start

            initial_retrieved_count = 0

            is_multi_document = (
                len(all_document_names) > 1
                and is_multi_document_question(
                    question,
                    all_document_names
                )
            )

            if is_multi_document:
                reranked_docs = balanced_multi_document_retrieval(
                    rewritten_question,
                    document_retrievers,
                    total_k=top_k
                )
            else:
                if selected_document == "All documents":

                    # Search the complete authorized document collection.
                    docs = retriever.invoke(rewritten_question)

                else:

                    # For a specific document, restrict retrieval to that document.
                    document_retriever = document_retrievers.get(
                        selected_document
                    )

                    if document_retriever is None:
                        docs = []
                    else:
                        docs = document_retriever.invoke(
                            rewritten_question
                        )

                initial_retrieved_count = len(docs)

                reranked_docs = rerank_documents(
                    question,
                    docs,
                    top_k=top_k
                )

                reranked_docs = prioritize_recent_documents(
                    reranked_docs
                )

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

            answer_start = time.time()
            result = llm.invoke(formatted_prompt)
            answer_time = time.time() - answer_start
            answer_text = result.content
            answer_length = len(answer_text)
            total_time = rewrite_time + retrieval_time + answer_time

            st.write("### 🔍 Retrieval Diagnostics")

            st.write(f"**Retrieval latency:** {retrieval_time:.3f} seconds")
            st.write(f"**Final chunks used:** {len(reranked_docs)}")

        st.session_state.chat_history.append(
    {
        "question": question,
        "answer": result.content,
        "sources": sources,
        "retrieval_time": retrieval_time,
        "rewrite_time": rewrite_time,
        "answer_time": answer_time,
        "total_time": total_time,
        "answer_length": answer_length,
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
                    st.write(f"**Query rewrite:** {chat['rewrite_time']:.2f} seconds")
                    st.write(f"**Retrieval + reranking:** {chat['retrieval_time']:.2f} seconds")
                    st.write(f"**Answer generation:** {chat['answer_time']:.2f} seconds")
                    st.write(f"**Total latency:** {chat['total_time']:.2f} seconds")
                    st.write(
                        f"**Answer length:** {chat['answer_length']} characters"
                    )
                    st.write("# 📄 Sources")

                    grouped_sources = defaultdict(list)

                    for item in chat["sources"]:
                        grouped_sources[item["source"]].append(item)

                    for source_file, items in grouped_sources.items():

                        st.markdown(f"**📄 {source_file}**")

                        for item in items:
                            file_extension = os.path.splitext(
                                source_file
                            )[1].lower()

                            if file_extension == ".pdf":
                                source_details = (
                                    f"Page {item['page']} · "
                                    f"Chunk {item['chunk_id']} · "
                                    f"Rerank score: "
                                    f"{item['score']:.2f}"
                                )
                            else:
                                source_details = (
                                    f"Chunk {item['chunk_id']} · "
                                    f"Rerank score: "
                                    f"{item['score']:.2f}"
                                )

                            st.write(
                                f"**[{item['citation_id']}]** "
                                f"{source_details}"
                            )

            st.divider()


    if st.button(":material/delete: Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()