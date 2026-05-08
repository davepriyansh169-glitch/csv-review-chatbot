import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- UI CONFIGURATION ---
st.set_page_config(page_title="AI Restaurant Agent", page_icon="🍴")
st.title("🍴 Restaurant Review Agent")
st.caption("Running Locally with Llama 3.2: 1B")

# --- 1. THE BRAIN: SETUP RAG PIPELINE ---
@st.cache_resource # This saves memory by only loading the AI once
def initialize_ai():
    # A. Load the Data
    df = pd.read_csv("realistic_restaurant_reviews.csv")
    documents = [
        Document(
            page_content=f"Title: {row['Title']} | Review: {row['Review']}", 
            metadata={"rating": row["Rating"]}
        ) for _, row in df.iterrows()
    ]
    
    # B. Create the Vector Store (The Search Engine)
    # Note: Using all-minilm for speed
    embeddings = OllamaEmbeddings(model="all-minilm")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Top 3 results
    
    # C. Setup the LLM (Using your 1B model)
    llm = ChatOllama(model="llama3.2:1b", temperature=0) 
    
    return retriever, llm

# Initialize
try:
    retriever, llm = initialize_ai()
except Exception as e:
    st.error("⚠️ Ollama Connection Error")
    st.write("1. Make sure the Ollama App is running in your taskbar.")
    st.write("2. Open terminal and run: `ollama pull llama3.2:1b` and `ollama pull all-minilm`")
    st.stop()

# --- 2. THE FACE: CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about the reviews (e.g., 'How is the service?')"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- THE RAG PROCESS ---
    with st.spinner("Searching reviews..."):
        # 1. Search for relevant reviews
        context_docs = retriever.invoke(prompt)
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        # 2. Build the Prompt
        template = """You are a helpful assistant. Use the following restaurant reviews to answer the question. 
        If you don't know the answer based on the reviews, just say you don't know.
        
        REVIEWS:
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