from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_API_KEY")


llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    task = "text-generation",
)


model = ChatHuggingFace(llm = llm)


def get_llm(document: str, query: str):
    prompt = PromptTemplate(
        template="Answer like a professional coder. Given the following document:\n{document}\nAnswer the following question:\n{query}",
        input_variables=["document", "query"]
    )
    
    # Format the prompt
    final_prompt = prompt.format(document=document, query=query)
    
    # Invoke the model
    response = model.invoke(final_prompt)
    
    return response.content



