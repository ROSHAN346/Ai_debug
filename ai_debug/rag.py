from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10
)



def split_document(text : str):

    chunk = splitter.split_text(text)
    return chunk