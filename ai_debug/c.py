from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("p.pdf")

docs = loader.load()

print(docs)