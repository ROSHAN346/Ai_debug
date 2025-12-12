import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_document(directory: str):
    
    files = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if (
            (file.endswith(".py") or file.endswith(".js")) 
            and os.path.isfile(os.path.join(directory, file))
        )
    ]

    documents = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10
    )

    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            text = file.read()
            chunks = splitter.split_text(text)

            for idx, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content = chunk, 
                        metadata = {
                            "source" : f, 
                            "id" : f"{os.path.basename(f)}_{idx}"
                        }
                    )
                )

            

    return documents
