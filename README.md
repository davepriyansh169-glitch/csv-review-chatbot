# AI CSV Data Agent

A generic local AI project that answers questions about any CSV file using a Retrieval-Augmented Generation (RAG) workflow. The app uses Streamlit for the interface, LangChain for orchestration, FAISS for retrieval, and Ollama for running local models.

## Project Description

This project allows users to upload any CSV file, converts the rows into searchable documents, and retrieves the most relevant data when a user asks a question. The selected context is then passed to a local Ollama model to generate an answer.

## Features

- Local AI question answering over any CSV data
- CSV file upload support
- Streamlit chat interface
- Retrieval-Augmented Generation (RAG) pipeline
- FAISS-based similarity search
- Ollama-powered local embeddings and chat model

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
|-- app.py
|-- sample_data.csv
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

## How to Run the Project

Run the Streamlit app:

```bash
streamlit run app.py
```

## Notes

- This project is designed to run locally.
- Ollama must be running before starting the app.
