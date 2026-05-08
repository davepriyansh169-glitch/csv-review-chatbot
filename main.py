from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

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

def format_reviews(docs):
    return "\n\n".join(f"- {d.page_content}" for d in docs)

while True:
    print("\n\n-------------------------------")
    question = input("ask your question (q to quit): ")
    print("\n\n")
    if question.strip().lower() == "q":
        break

    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} reviews.")
    reviews_text = format_reviews(docs)
    print("Generating answer...\n")
    result = chain.invoke({"reviews": reviews_text, "question": question})
    print("--- RESPONSE ---")
    print(result)

