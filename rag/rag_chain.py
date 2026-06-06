from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Load vectorstore
db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Load LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

while True:
    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

    # Retrieve relevant chunks
    docs = db.similarity_search(question, k=4)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the provided context.

If the answer is not found in the context, reply:
"I could not find that information in the document."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)

    print("\nSources Retrieved:")
    for i, doc in enumerate(docs):
        print(f"\n--- Source {i+1} ---")
        print(doc.page_content[:300])