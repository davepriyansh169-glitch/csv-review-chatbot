from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="tinyllama")

template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews:
{reviews}

Here is the question to answer:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

reviews_text = "The pepperoni pizza was amazing! Best I've ever had."
question = "What is the best pizza?"

print("Running manual chain test...")
try:
    result = chain.invoke({"reviews": reviews_text, "question": question})
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
