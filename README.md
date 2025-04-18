# Project Neocortex: Personal Knowledge RAG System

**Status:** Core RAG pipeline (Ingestion & Query via CLI) functional.

## Overview

Project Neocortex is a Retrieval-Augmented Generation (RAG) system designed to interact with a personal knowledge base built primarily from Readwise highlights. It allows a user to ask questions in natural language and receive context-aware answers generated by a local Large Language Model (LLM), based on relevant retrieved highlights.

This system runs locally, leveraging quantized LLMs (via Ollama) and a local vector store (ChromaDB) for efficiency and privacy.

## Features (Implemented V0.1)

* **Readwise Integration:** Fetches full export data (sources and nested highlights) from the Readwise API, handling pagination and basic rate limiting.
* **Data Processing:** Converts raw Readwise data into LlamaIndex `Document` objects, extracting metadata and preparing text content.
* **Vector Store:** Uses ChromaDB for persistent local storage of document embeddings and metadata (`./chroma_db_store`).
* **Embedding:** Uses the `sentence-transformers/all-MiniLM-L6-v2` model (via HuggingFace & LlamaIndex) to generate text embeddings.
* **LLM Interaction:** Connects to a locally running Ollama instance to use the `mistral` model (via LlamaIndex `Ollama` integration).
* **RAG Pipeline:** Implements a retrieval pipeline using LlamaIndex `VectorStoreIndex` and `QueryEngine` to find relevant documents (currently top 5) based on semantic similarity to the query and generate an answer using the LLM.
* **Command-Line Interface (CLI):** Provides an interactive CLI (`src/main.py`) for asking questions and receiving answers.

## Tech Stack

* **Language:** Python 3.1x
* **Core Libraries:**
    * `llama-index` (Core framework, integrations)
    * `llama-index-llms-ollama`
    * `llama-index-vector-stores-chroma`
    * `llama-index-embeddings-huggingface`
    * `chromadb` (Vector database client)
    * `sentence-transformers` (Underlying embedding model library)
    * `requests` (For Readwise API calls)
    * `python-dotenv` (For environment variables)
* **LLM Runner:** Ollama (Running `mistral` model)
* **Environment Management:** `venv` + `pip`

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    # Navigate to your ai_ml directory (e.g., ~/project_n88/ai_ml/)
    # git clone <your-repo-url> # If cloning from remote
    # cd project_rag_neocortex
    ```
2.  **Install Ollama:** Download and install Ollama for macOS from [ollama.com](https://ollama.com/download). Launch the application.
3.  **Pull Mistral Model:** Open your terminal and run:
    ```bash
    ollama pull mistral
    ```
4.  **Set up Python Environment:** (Run from the `project_rag_neocortex` directory)
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
6.  **Configure API Key:**
    * Rename `.env.example` to `.env`: `mv .env.example .env`
    * Edit the `.env` file and add your Readwise API token:
        ```env
        READWISE_API_KEY=YOUR_READWISE_API_TOKEN_HERE
        ```
7.  **Initial Data Ingestion:**
    * This step fetches all your Readwise data, processes it, generates embeddings, and stores it in ChromaDB. **This can take a significant amount of time.**
    * Make sure your `venv` is active and Ollama is running. Run from the project root (`project_rag_neocortex`):
        ```bash
        python -m src.data_ingestion.ingest
        ```
    * This will create the `./chroma_db_store` directory.

## Usage (CLI)

1.  **Ensure Ollama is running.**
2.  **Activate the virtual environment:** (From the `project_rag_neocortex` directory)
    ```bash
    source venv/bin/activate
    ```
3.  **Run the main CLI script:**
    ```bash
    python -m src.main
    ```
4.  The script will display a welcome message and prompt you for your query.
5.  Type your question and press Enter.
6.  The system will process the query, retrieve relevant highlights, generate a response using Mistral, and print the answer.
7.  Type `quit` or `exit` to stop the CLI.

## Project Structure

```text
project_rag_neocortex/
│
├── .gitignore
├── README.md             # This file
├── CHANGELOG.md
├── PRD.md
├── TDD.md
├── requirements.txt
├── .env                  # Local environment variables (gitignored)
├── .env.example
├── venv/                 # Python virtual environment (gitignored)
├── chroma_db_store/      # Persistent ChromaDB data (gitignored)
│
├── src/                  # Source code package
│   ├── __init__.py
│   ├── main.py           # CLI entry point
│   │
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── readwise_client.py
│   │   └── ingest.py
│   │
│   ├── vector_store/
│   │   ├── __init__.py
│   │   └── store_manager.py
│   │
│   ├── rag_pipeline/
│   │   ├── __init__.py
│   │   └── pipeline.py
│   │
│   └── llm_interface/
│       ├── __init__.py
│       └── generator.py
│
└── notebooks/            # Jupyter notebooks for experimentation (Optional)
    └── exploration.ipynb
