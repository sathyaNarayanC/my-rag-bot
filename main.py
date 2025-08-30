"""
# My RAG Bot

This project is a Retrieval-Augmented Generation (RAG) bot that leverages LangChain, Chroma vector database, and HuggingFace models to answer questions based on your own documents (CSV, PDF, TXT). It loads, chunks, embeds, and indexes your data, then uses a local LLM to answer queries with context retrieved from your documents.

## Features
- Ingests CSV, PDF, and TXT files
- Chunks documents to avoid token overflow
- Embeds and stores chunks in a Chroma vector database
- Uses HuggingFace LLM (e.g., Flan-T5) for answer generation
- Retrieval-augmented QA pipeline

"""

import os
import pandas as pd

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- Config ---
CSV_PATH = "data/ToTo.csv"
PDF_PATH = "data/test.pdf"
TEXT_PATH = "data/sg60-national-day.txt"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "my_docs"
CHUNK_SIZE = 150       # smaller to avoid token overflow
CHUNK_OVERLAP = 100
TOP_K = 3              # reduce top-k to avoid token overflow


# ---------------- DOCUMENT INGEST ----------------
def ingest_csv(file_path):
    df = pd.read_csv(file_path)
    docs = []
    for _, row in df.iterrows():
        # Concatenate columns with header
        text = "; ".join([f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])])
        docs.append(Document(page_content=text))
    return docs


def ingest_docs(csv_path=None, pdf_path=None, text_path=None):
    docs = []
    if os.path.exists(csv_path):
        docs.extend(ingest_csv(csv_path))
    if os.path.exists(pdf_path):
        docs.extend(PyPDFLoader(pdf_path).load())
    if os.path.exists(text_path):
        docs.extend(TextLoader(text_path).load())
    return docs

# ---------------- CHUNKING ----------------
def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

# ---------------- VECTOR DB ----------------
# Clear and rebuild Chroma DB
def rebuild_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

    # Initialize Chroma with the collection name
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    # Reset this collection (no arguments needed)
    print(f"Resetting collection '{COLLECTION_NAME}'...")
    vectordb.reset_collection()

    # Insert new chunks
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR
    )

    return vectordb

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    return vectordb

# ---------------- HUGGINGFACE LLM ----------------
def load_hf_llm(model_name="google/flan-t5-large"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        truncation=True  # prevent token overflow
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# ---------------- RAG ANSWER ----------------
def rag_answer(retriever, llm, query, top_k=TOP_K):
    # Updated from deprecated get_relevant_documents → invoke
    # docs = retriever.invoke(query)[:top_k]
    # context = "\n\n---\n\n".join([d.page_content for d in docs]) or "No context found."
    # prompt = f"Answer the question using ONLY the context below:\n\nCONTEXT:\n{context}\n\nQUESTION: {query}"
    #
    #
    #
    # qa = RetrievalQA.from_chain_type(retriever=retriever, llm=llm, chain_type="stuff", prompt=prompt)
    # return qa.invoke()
    # return llm.invoke(prompt)

    docs = retriever.invoke(query)[:top_k]
    print(f"Retrieving {len(docs)} documents...")
    context = "\n\n---\n\n".join([d.page_content for d in docs]) or "No context found."
    print(f"Context: {context}")
    print("==========================================================")
    prompt = f"Answer the question using ONLY the context below:\n\nCONTEXT:\n{context}\n\nQUESTION: {query}"
    print(f"\n{prompt}")
    return llm.invoke(prompt)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    docs = ingest_docs(CSV_PATH, PDF_PATH, TEXT_PATH)
    print(f"Total documents loaded: {len(docs)}")

    chunks = chunk_docs(docs)
    print(f"Total chunks created: {len(chunks)}")

    vectordb = rebuild_vectorstore(chunks)
    # vectordb = build_vectorstore(chunks)
    retriever = vectordb.as_retriever()

    llm = load_hf_llm("google/flan-t5-large")  # can switch to larger model if needed

    # ================ Sample queries =======================
    # query = "Summarize the Fireworks for the national day.?."
    # query = "What is the Chronological order of the national day in singapore ?."
    # query = "What is the Theme of Singapore national day ?"
    # query = "What is the Theme of Singapore national day happened in 2025 ?"
    # query = "What is the highest Draw Number in descending order?"
    # =======================================================
    query = "How many high and how many low in Draw 4099?"
    print("\n ================ RAG Query ================")
    print(query)
    print("\n ===========================================")
    answer = rag_answer(retriever, llm, query)
    print("\n================ RAG Answer ================")
    print(answer)
    print("\n ===========================================")
