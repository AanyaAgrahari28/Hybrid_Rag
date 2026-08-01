import streamlit as st
import tempfile

from rag_utils import build_rag

st.set_page_config(page_title="Hybrid RAG")

st.title("📄 Hybrid RAG")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    if "qa_chain" not in st.session_state:

        with st.spinner("Processing PDF..."):
            st.session_state.qa_chain = build_rag(pdf_path)

        st.success("PDF processed successfully!")

    question = st.text_input("Ask a question")

    if st.button("Ask"):

        st.write("Button clicked!")

        if question:

            st.write("Question:", question)

            with st.spinner("Generating answer..."):

                result = st.session_state.qa_chain.invoke(
                    {"query": question}
                )

            st.subheader("Answer")
            st.write(result["result"])