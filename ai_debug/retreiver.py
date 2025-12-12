from langchain_community.retrievers import SVMRetriever
from langchain_huggingface import HuggingFaceEndpoint


# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7,
)


def generate_retriever(vectorstore, retriever_type: str):
    
    

    if retriever_type == "svm":
        # Extract documents from FAISS
        documents = list(vectorstore.docstore._dict.values())

        return SVMRetriever.from_documents(
            documents=documents,
            embeddings=vectorstore.embeddings,
        )

    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
