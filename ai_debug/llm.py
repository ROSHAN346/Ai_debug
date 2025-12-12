from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from typing import List , TypedDict 
# from langchain_community.prompts import PromptTemplate

llm = HuggingFaceEndpoint(
    repo_id= "",
    method = "text-generation"
)

model = ChatHuggingFace(llm = llm)




