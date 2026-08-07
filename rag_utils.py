# import dependencies
import sys
import langchain
import uuid

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

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Cross Encoder loaded successfully.")

def rerank_documents(question, documents, top_k=3):

    pairs = [
        (question, doc.page_content)
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for score, doc in ranked_docs[:top_k]]

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
                "Unable to process this PDF. It may be protected, corrupted, or unsupported."
            )
    if not documents:
        raise ValueError("No text could be extracted from this PDF.")
    
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

    collection_name = f"pdf_{uuid.uuid4().hex}"

    vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
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

    local_llm =ChatOllama(
        model = 'gemma4:e4b',
        max_tokens=512,
        temperature=0.3
    )

    prompt_template = """
    You are an Enterprise Knowledge Assistant.

    Your task is to answer questions ONLY using the retrieved context from the uploaded document.

    Rules:
    1. Never use outside knowledge.
    2. Never hallucinate or guess.
    3. If the answer is not found in the retrieved context, reply:
    "I don't know based on the uploaded document."
    4. Prefer precise information over lengthy explanations.
    5. Preserve names, numbers, dates, formulas, technical terms, and definitions exactly as written.
    6. Remove duplicate information.
    7. Keep responses clear, professional, and well-structured.

    Response Style:

    • If the user asks to summarize:
    - Give a concise summary in 5-8 bullet points.

    • If the user asks for important points, key points, highlights, or takeaways:
    - Return the most important facts as bullet points.

    • If the user asks to explain:
    - Explain step by step with headings.

    • If the user asks "what is" or "define":
    - Give a short definition first, then a brief explanation.

    • If the user asks to compare:
    - Use a markdown table whenever possible.

    • If the user asks to list:
    - Return a clean bullet list.

    • If the user asks for advantages/disadvantages:
    - Use separate headings.

    • If the user asks for steps or process:
    - Use a numbered list.

    • If the answer contains multiple topics:
    - Organize using headings and bullet points.

    • Keep answers concise unless the user explicitly asks for a detailed explanation.

    Retrieved Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate.from_template(prompt_template)

    return {
    "retriever": hybrid_retriever,
    "llm": local_llm,
    "prompt": prompt,
}

