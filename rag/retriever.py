from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Load FAISS vectorstore
db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Test query
query = "What is the name of the university?"

results = db.similarity_search(
    query,
    k=3
)

print("\nRetrieved Chunks:\n")

for i, doc in enumerate(results):
    print(f"\n----- Chunk {i+1} -----\n")
    print(doc.page_content[:1000])