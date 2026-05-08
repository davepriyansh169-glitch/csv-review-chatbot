from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import pandas as pd


df = pd.read_csv("realistic_restaurant_reviews.csv")


documents = [
    Document(
        page_content=row["Title"] + " " + row["Review"],
        metadata={"rating": row["Rating"], "date": row["Date"]}
    )
    for _, row in df.iterrows()
]


embeddings = OllamaEmbeddings(model="all-minilm") 


vectorstore = FAISS.from_documents(documents, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})