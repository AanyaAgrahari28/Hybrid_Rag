# import dependencies
import sys
import langchain
import uuid
import os
import json

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
from datetime import datetime

from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

KNOWLEDGE_BASE_DIR = "knowledge_base"
DOCUMENT_REGISTRY = os.path.join(
    KNOWLEDGE_BASE_DIR,
    "documents.json"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Cross Encoder loaded successfully.")

def rerank_documents(question, documents, top_k=6):

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

    final_docs = []

    for score, doc in ranked_docs[:top_k]:
        doc.metadata["rerank_score"] = float(score)
        final_docs.append(doc)

    return final_docs

def balanced_multi_document_retrieval(
    question,
    document_retrievers,
    total_k=6,
    per_document_k=6
):
    all_docs = []

    # Retrieve from each document separately
    for document_retriever in document_retrievers.values():
        all_docs.extend(
            document_retriever.invoke(question)
        )

    # Remove duplicate chunks using a stable chunk identity
    unique_docs = {}

    for doc in all_docs:
        source_file = str(
            doc.metadata.get("source_file", "")
        ).strip()

        chunk_id = str(
            doc.metadata.get("chunk_id", "")
        ).strip()

        key = (source_file, chunk_id)

        if key not in unique_docs:
            unique_docs[key] = doc

    docs = list(unique_docs.values())

    # Rerank all unique candidates
    ranked_docs = rerank_documents(
        question,
        docs,
        top_k=len(docs)
    )

    # Select at least one unique chunk from each document
    selected_docs = []
    selected_keys = set()

    for source_file in document_retrievers:

        for doc in ranked_docs:

            doc_source = str(
                doc.metadata.get("source_file", "")
            ).strip()

            chunk_id = str(
                doc.metadata.get("chunk_id", "")
            ).strip()

            key = (doc_source, chunk_id)

            if (
                doc_source == source_file
                and key not in selected_keys
            ):
                selected_docs.append(doc)
                selected_keys.add(key)
                break

    # Fill remaining slots with highest-ranked UNIQUE chunks
    for doc in ranked_docs:

        if len(selected_docs) >= total_k:
            break

        doc_source = str(
            doc.metadata.get("source_file", "")
        ).strip()

        chunk_id = str(
            doc.metadata.get("chunk_id", "")
        ).strip()

        key = (doc_source, chunk_id)

        if key not in selected_keys:
            selected_docs.append(doc)
            selected_keys.add(key)

    return selected_docs[:total_k]

def filter_documents_by_metadata(documents, source_file=None, page_number=None):
    """
    Filter retrieved documents using metadata.

    Args:
        documents: List of LangChain Documents
        source_file: Optional PDF filename to filter by
        page_number: Optional page number to filter by

    Returns:
        Filtered list of documents
    """

    filtered_docs = documents

    if source_file:
        filtered_docs = [
            doc for doc in filtered_docs
            if doc.metadata.get("source_file") == source_file
        ]

    if page_number is not None:
        filtered_docs = [
            doc for doc in filtered_docs
            if doc.metadata.get("page") == page_number
        ]

    return filtered_docs

def prioritize_recent_documents(documents, recent_boost=0.15):
    """
    Give a small ranking boost to documents uploaded more recently.
    """

    now = datetime.now()

    for doc in documents:
        upload_time = doc.metadata.get("upload_time")

        if upload_time:
            try:
                upload_time = datetime.fromisoformat(upload_time)

                age_days = max((now - upload_time).total_seconds() / 86400, 0)

                # Smaller age = larger boost
                boost = recent_boost / (1 + age_days)

                current_score = doc.metadata.get("rerank_score", 0)
                doc.metadata["rerank_score"] = current_score + boost

            except (ValueError, TypeError):
                pass

    return sorted(
        documents,
        key=lambda doc: doc.metadata.get("rerank_score", 0),
        reverse=True
    )

def rewrite_query(question, llm):
    """
    Rewrite a user's question into a clear, retrieval-friendly query.
    """

    rewrite_prompt = f"""
You are a query rewriting assistant for an enterprise document search system.

Rewrite the user's question into a clear and precise search query.

Rules:
- Preserve the original meaning.
- Do not add information that is not present in the question.
- Resolve obvious conversational wording.
- Keep important names, terms, numbers, and dates.
- Return ONLY the rewritten query.
- Do not answer the question.

User question:
{question}

Rewritten query:
"""

    response = llm.invoke(rewrite_prompt)

    return response.content.strip()

def load_document(file_path, file_name):

    extension = file_name.lower().rsplit(".", 1)[-1]

    if extension == "pdf":
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception:
            loader = PyMuPDFLoader(file_path)
            return loader.load()

    elif extension == "docx":
        loader = Docx2txtLoader(file_path)
        return loader.load()

    elif extension in ["txt", "md"]:
        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )
        return loader.load()

    else:
        raise ValueError(
            f"Unsupported file type: .{extension}"
        )

def load_document_registry():

    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

    if not os.path.exists(DOCUMENT_REGISTRY):
        return []

    with open(
        DOCUMENT_REGISTRY,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)

def save_document_registry(documents):

    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

    with open(
        DOCUMENT_REGISTRY,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            documents,
            f,
            indent=4
        )

def build_rag(document_paths): 
    
    # Load documents and split into chunks
    # Load documents from all uploaded PDFs
    all_documents = []

    for document_path, document_name in document_paths:
        upload_time = datetime.now().isoformat()

        try:
            documents = load_document(
                document_path,
                document_name
            )

        except Exception:
            raise ValueError(
                f"Unable to process '{document_name}'. "
                "The file may be corrupted or unsupported."
            )

        if not documents:
            raise ValueError(
                f"No text could be extracted from '{document_name}'."
            )

        # Preserve the original PDF name for future citations
        for doc in documents:
            doc.metadata["source_file"] = document_name
            doc.metadata["upload_time"] = upload_time

        all_documents.extend(documents)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=75)
    docs = text_splitter.split_documents(all_documents)
    registry = load_document_registry()

    for document_name in set(
        doc.metadata["source_file"] for doc in docs
    ):
        document_chunks = [
            doc for doc in docs
            if doc.metadata["source_file"] == document_name
        ]

        existing = next(
            (
                item for item in registry
                if item["filename"] == document_name
            ),
            None
        )

        if existing is None:
            registry.append({
                "filename": document_name,
                "file_type": document_name.rsplit(".", 1)[-1],
                "upload_time": datetime.now().isoformat(),
                "chunks": len(document_chunks)
            })

    save_document_registry(registry)


    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = i + 1

    print("Documents:", len(all_documents))
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

    print("Creating Chroma...")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=os.path.join(
        KNOWLEDGE_BASE_DIR,
        "chroma_db"
    ),
        collection_name="enterprise_documents",
    )
    print("Chroma created successfully.")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 10
    hybrid_retriever = EnsembleRetriever(
        retrievers=[retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    document_retrievers = {}

    for source_file in set(doc.metadata["source_file"] for doc in docs):
        source_docs = [
            doc for doc in docs
            if doc.metadata["source_file"] == source_file
        ]

        source_bm25 = BM25Retriever.from_documents(source_docs)
        source_bm25.k = 6

        source_vectorstore = Chroma.from_documents(
            documents=source_docs,
            embedding=embeddings
        )

        source_vector = source_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )

        document_retrievers[source_file] = EnsembleRetriever(
            retrievers=[source_vector, source_bm25],
            weights=[0.5, 0.5]
        )

    local_llm =ChatOllama(
        model = 'gemma4:e4b',
        max_tokens=512,
        temperature=0.3
    )

    prompt_template = """
    You are an Enterprise Knowledge Assistant.

    Your task is to answer questions ONLY using the retrieved context from the uploaded documents.

    Rules:
    1. Never use outside knowledge.
    2. Never hallucinate or guess.
    3. If the answer is not found in the retrieved context, reply:
    "I don't know based on the uploaded documents."
    4. Prefer precise information over lengthy explanations.
    5. Preserve names, numbers, dates, formulas, technical terms, and definitions exactly as written.
    6. Remove duplicate information.
    7. Keep responses clear, professional, and well-structured.
    8. Cite factual statements using [1], [2], [3], etc.
    9. Each citation number corresponds to the [SOURCE N] section in the retrieved context.
    10. Never invent a citation number.
    11. Place citations immediately after the statement they support.
    12. If multiple sources support a statement, cite all relevant sources.

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
    "document_retrievers": document_retrievers,
    "llm": local_llm,
    "prompt": prompt,
}

