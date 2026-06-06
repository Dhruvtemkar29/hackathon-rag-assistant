from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

response = llm.invoke("What is Artificial Intelligence?")

print(response.content)