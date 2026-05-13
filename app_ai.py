import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- UI CONFIGURATION ---
st.set_page_config(page_title="AI CSV Agent", page_icon="📄")
st.title("📄 AI CSV Data Agent")
st.caption("Running Locally with Ollama")

# --- 1. THE BRAIN: SETUP RAG PIPELINE ---

@st.cache_resource
def load_llm():
    return ChatOllama(model="llama3.2:1b", temperature=0)

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="all-minilm")

@st.cache_resource(show_spinner="Processing CSV and building search index...")
def get_retriever(df):
    # Combine all columns into a single string for each row
    documents = []
    for _, row in df.iterrows():
        # Filtering out null values and joining
        content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        documents.append(Document(page_content=content))
    
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# --- FILE UPLOADER ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

# Load Data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded: {uploaded_file.name}")
else:
    # Try to load default file if it exists
    try:
        df = pd.read_csv("realistic_restaurant_reviews.csv")
        st.sidebar.info("Using default restaurant reviews CSV.")
    except FileNotFoundError:
        st.warning("Please upload a CSV file to begin.")
        st.stop()

# Initialize AI Components
try:
    llm = load_llm()
    retriever = get_retriever(df)
except Exception as e:
    st.error("⚠️ Ollama Connection Error")
    st.write("1. Make sure the Ollama App is running.")
    st.write("2. Open terminal and run: `ollama pull llama3.2:1b` and `ollama pull all-minilm`")
    # st.exception(e) # For debugging
    st.stop()

# --- 2. THE FACE: CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your data..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- THE RAG PROCESS ---
    with st.spinner("Analyzing data..."):
        try:
            # 1. Search for relevant context
            context_docs = retriever.invoke(prompt)
            context_text = "\n\n".join([doc.page_content for doc in context_docs])

            # 2. Build the Prompt
            template = """You are a helpful assistant. Use the following pieces of context from a CSV file to answer the question.
            If you don't know the answer based on the context, just say you don't know.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            ANSWER:"""

            prompt_template = ChatPromptTemplate.from_template(template)

            # 3. Generate Answer
            chain = prompt_template | llm
            response = chain.invoke({"context": context_text, "question": prompt})

            # Show Assistant response
            with st.chat_message("assistant"):
                st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
        except Exception as e:
            st.error(f"An error occurred: {e}")
