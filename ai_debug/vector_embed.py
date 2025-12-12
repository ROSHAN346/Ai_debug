import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

FAISS_PATH = "./faiss_store"


def create_faiss(documents):
    # Create FAISS index
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model
    )

    # Persist FAISS index manually
    vectorstore.save_local(FAISS_PATH)
    
    return vectorstore

def load_faiss():

    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    return vectorstore
