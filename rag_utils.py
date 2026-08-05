# import dependencies
import sys
import langchain

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama

from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

def build_rag(pdf_path): 
    
    # Load documents and split into chunks
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    except Exception:

        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()

        except Exception:
            raise ValueError(
                "This PDF is encrypted, corrupted, or unsupported."
            )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=75)
    docs = text_splitter.split_documents(documents)
    print("Documents:", len(documents))
    print("Chunks:", len(docs))

    if len(docs) == 0:
        raise ValueError("No text could be extracted from this PDF.")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        dimensions=768
    )
    print("First chunk:")
    print(docs[0].page_content[:500])
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db_Vermaanant",
        collection_name="ollama_test"
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5
    hybrid_retriever = EnsembleRetriever(
        retrievers=[retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    retriever.invoke("What is the main topic of the paper?")
    local_llm =ChatOllama(
        model = 'gemma4:e4b',
        max_tokens=512,
        temperature=0.3
    )
    prompt_template = """
        You are a helpful assistant.

        Answer ONLY using the retrieved context.

        Keep the answer concise.

        If the user asks for a summary, answer in 5-8 bullet points.

        If the answer is not present, say "I don't know."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

    prompt = PromptTemplate.from_template(prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm,
        chain_type = "stuff",
        retriever=hybrid_retriever,
        verbose=True)
    return qa_chain

