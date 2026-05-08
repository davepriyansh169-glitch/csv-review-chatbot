from langchain_ollama.llms import OllamaLLM
model = OllamaLLM(model="tinyllama")
print("Testing tinyllama...")
try:
    response = model.invoke("say hi")
    print(f"Response: {response}")
except Exception as e:
    print(f"Error: {e}")
