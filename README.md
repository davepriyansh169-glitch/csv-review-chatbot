# AI Restaurant Review Agent

A beginner-friendly local AI project that answers questions about restaurant reviews using a Retrieval-Augmented Generation (RAG) workflow. The app uses Streamlit for the interface, LangChain for orchestration, FAISS for retrieval, and Ollama for running local models.

## Project Description

This project loads a restaurant review dataset, converts the reviews into searchable documents, and retrieves the most relevant reviews when a user asks a question. The selected review context is then passed to a local Ollama model to generate an answer.

## Features

- Local AI question answering over restaurant review data
- Streamlit chat interface
- Retrieval-Augmented Generation (RAG) pipeline
- FAISS-based similarity search
- Ollama-powered local embeddings and chat model
- Simple Python test scripts for local model checks

## Tech Stack

- Python
- Streamlit
- Pandas
- LangChain
- FAISS
- Ollama

## Project Structure

```text
.
|-- app_ai.py
|-- main.py
|-- vector.py
|-- realistic_restaurant_reviews.csv
|-- test_chain.py
|-- test_embeddings.py
|-- test_ollama.py
|-- requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Ollama Setup

1. Install Ollama from [https://ollama.com/](https://ollama.com/).
2. Start the Ollama app or Ollama service on your computer.
3. Pull the models used by the Streamlit app:

```bash
ollama pull llama3.2:1b
ollama pull all-minilm
```

Optional note: some test scripts in this repository reference `tinyllama` and `mxbai-embed-large`. If you want to run those test files too, pull those models separately:

```bash
ollama pull tinyllama
ollama pull mxbai-embed-large
```

## How to Run the Project

Run the Streamlit app:

```bash
streamlit run app_ai.py
```

If you want to try the terminal-based script:

```bash
python main.py
```

## Future Improvements

- Add better error handling and startup checks
- Save vector indexes locally for faster startup
- Add filtering by rating or review date
- Improve prompt design for more consistent answers
- Add automated tests for the RAG flow

## Notes

- This project is designed to run locally.
- Ollama must be running before starting the app.
- Large model files are intentionally excluded from Git tracking.
