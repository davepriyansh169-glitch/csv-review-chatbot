from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
print("Testing mxbai-embed-large...")
try:
    vector = embeddings.embed_query("test")
    print(f"Vector size: {len(vector)}")
except Exception as e:
    print(f"Error: {e}")
