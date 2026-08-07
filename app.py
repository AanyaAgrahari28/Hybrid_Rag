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

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    if (
        "qa_chain" not in st.session_state
        or st.session_state.current_pdf != uploaded_file.name
    ):

        with st.spinner("Processing PDF..."):
            try:
                st.session_state.qa_chain = build_rag(pdf_path)

            except Exception as e:
                st.error(str(e))
                st.stop()
        st.session_state.current_pdf = uploaded_file.name
        st.session_state.chat_history = []

        st.success("Document processed successfully.")

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

            formatted_prompt = prompt.format(
                context=context,
                question=question
            )

            result = llm.invoke(formatted_prompt)

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": result.content
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

            st.divider()


    if st.button(":material/delete: Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()