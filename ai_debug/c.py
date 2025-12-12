from splitter import split_document 
from dotenv import load_dotenv
from vector_embed import create_faiss 
from langchain_huggingface import HuggingFaceEmbeddings
from retreiver import generate_retriever
from llm import get_llm
load_dotenv()

directory = "C:/AIDebug/Ai_debug/ai_debug"
documents = split_document(directory)

vectorstore = create_faiss(documents)

query = input("Enter your query: ")

operation = ["svm"]

docs = ""

for op in operation:
    retriever = generate_retriever(vectorstore, retriever_type=op)

    m = retriever.invoke(query)
    for d in m:
        docs = "".join([d.page_content , "and its location is " , d.metadata["source"] , "\n"])


print(type(docs))
result = get_llm(docs , query)

print("Final Result: ", result)
# print(vectorstore)



