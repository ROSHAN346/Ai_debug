from splitter import split_document 
from dotenv import load_dotenv
from vector_embed import create_faiss 
import os
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()


directory = "C:/Users/MSI GAMING/OneDrive/Desktop/roshan/Project/ai_debug/ai_debug"
# Split documents in the directory
documents = split_document(directory)



# for doc in documents:
#     print(doc)
#     print("-----")

# print(documents[0].page_content)
# print(documents[0].metadata["source"])

# Initialize HuggingFace Embedding Model
vectorstore = create_faiss(documents)

results = vectorstore.similarity_search("I want to find the where we generate the documentation", k=2)
print(results)


# print(vectorstore)


# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# # Directory to store Chroma DB
# CHROMA_PATH = "./chroma_store"
