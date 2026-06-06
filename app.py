import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
from memory.db import save_message

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 2rem;
}

.chat-title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    margin-bottom:10px;
}

.chat-subtitle {
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("⚙️ Control Panel")

    st.success("RAG System Active")

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📄 Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())

            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)
        @st.cache_resource
        def load_embeddings():
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5"
            )

        embeddings = load_embeddings()

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )
        
        st.session_state["db"] = vectorstore
        

        st.sidebar.success(
            f"Pages Loaded: {len(docs)}"
        )

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        if "db" in st.session_state:
            del st.session_state["db"]

        st.rerun()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="chat-title">🤖 BrainVault</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="chat-subtitle">Ask questions from your indexed documents</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD COMPONENTS
# --------------------------------------------------

@st.cache_resource
def load_llm():

    return ChatGroq(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )

try:

    if "db" not in st.session_state:
        st.session_state["db"] = None
    llm = load_llm()

except Exception as e:

    st.error(f"Startup Error: {e}")
    st.stop()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "session_id" not in st.session_state:

    st.session_state.session_id = str(
        uuid.uuid4()
    )
    
    

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
history = "\n".join(
    [
        f"{msg['role']}: {msg['content']}"
        for msg in st.session_state.messages[-6:]
    ]
)

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

db = st.session_state.get("db")

if db is None:

    st.info(
        "📄 Please upload a PDF to begin chatting."
    )

    st.stop()

question = st.chat_input(
    "Ask a question about the document..."
)

if question:

    # User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )
    
    save_message(
        st.session_state.session_id,
        "user",
        question
        )  
    
    st.rerun()
    

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            try:

                docs = db.similarity_search(
                    question,
                    k=2
                )

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                prompt = f"""
You are an intelligent document assistant.

Use:
1. Conversation History
2. Retrieved Document Context

Rules:
- Answer from the document context whenever possible.
- Use conversation history to understand follow-up questions.
- If the answer cannot be found in the document context, say:
"I could not find that information in the document."
- Keep answers concise and accurate.

Conversation History:
{history}

Context:
{context}

Question:
{question}
"""

                response = llm.invoke(prompt)

                answer = response.content

                st.markdown(answer)

                with st.expander("📚 Sources Used"):

                    for i, doc in enumerate(docs):

                        st.markdown(
                            f"### Source {i+1}"
                        )

                        st.write(
                            doc.page_content[:700]
                        )

                        st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )
                
                save_message(
                    st.session_state.session_id,
                    "assistant",
                    answer
                )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )