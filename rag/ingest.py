from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("docs/Christ_syllabus.pdf")
docs = loader.load()

print(f"Pages Loaded: {len(docs)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

print(f"Total Chunks Created: {len(chunks)}")

print(chunks[0].page_content[:500])