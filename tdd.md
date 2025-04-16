graph LR
    subgraph User Interaction
        CLI[CLI (main.py)]
    end

    subgraph Core System
        A[main.py] --> B(RAG Pipeline - pipeline.py);
        B --> C{Vector Store Manager - store_manager.py};
        B --> D(LLM Interface - generator.py);
        C --> E[(ChromaDB Store)];
        D --> F[(Ollama Server + Mistral)];
    end

    subgraph Data Management / Ingestion Flow
        H(Ingestion Script - ingest.py) --> G(Readwise Client - readwise_client.py);
        H --> C;
        RWAPI(Readwise API) --> G;
        subgraph Manual Trigger
            I[User runs ingest.py] --> H;
        end
    end

    CLI --> B;


    style E fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#ccf,stroke:#333,stroke-width:2px

**Components:**
1 Readwise Client (src/data_ingestion/readwise_client.py): Handles authenticated communication with the Readwise API (/export endpoint). Fetches full export data, handling pagination and basic rate limiting. Provides get_readwise_token and fetch_all_highlights.
2 Ingestion Script (src/data_ingestion/ingest.py): Orchestrates data fetching (via readwise_client) and processing. Converts raw Readwise export data (nested list of sources and highlights) into LlamaIndex Document objects, mapping relevant fields to text content and metadata (tags converted to comma-separated strings). Provides process_highlights_to_documents. When run directly, it calls the store_manager to save the documents.
3 Vector Store Manager (src/vector_store/store_manager.py): Manages interaction with the ChromaDB vector store. Initializes the persistent client (./chroma_db_store), gets/creates the collection (readwise_highlights), initializes the embedding model (HuggingFaceEmbedding), and provides add_documents_to_store which uses VectorStoreIndex.from_documents to handle embedding and storage. Provides get_vector_store and get_embedding_model.
4 Vector DB (./chroma_db_store): Persistent ChromaDB directory storing embeddings, text, and metadata.
5 LLM Interface (src/llm_interface/generator.py): Interface for interacting with the local Ollama LLM. Initializes the LlamaIndex Ollama wrapper for the specified model (mistral). Provides get_llm.
**6** **Local LLM (Ollama + Mistral):** The Ollama application running the mistral model locally.
7 RAG Pipeline (src/rag_pipeline/pipeline.py): Coordinates the query process. Initializes components using Settings, loads the VectorStoreIndex from ChromaDB, creates a QueryEngine (index.as_query_engine), and provides query_knowledge_base function. Handles logging of retrieved nodes.
8 Main Entry Point (src/main.py): Provides a simple interactive command-line interface (CLI). Imports query_knowledge_base, takes user input in a loop, calls the query function, and prints the response. Executed via python -m src.main.

⠀**3.1 Directory Structure**
project_rag_neocortex/ (Root: ~/project_n88/ai_ml/project_rag_neocortex/)
│
├── .gitignore
├── README.md
├── CHANGELOG.md
├── PRD.md
├── TDD.md                # This document
├── requirements.txt
├── .env                  # Local environment variables (gitignored)
├── .env.example
├── venv/                 # Python virtual environment (gitignored)
├── chroma_db_store/      # Persistent ChromaDB data (gitignored)
│
├── src/                  # Main source code package
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

## 4. Technology Choices
* **Programming Language:** Python 3.1x (e.g., 3.13 as seen in tracebacks)
* **Core Framework:** **LlamaIndex** (Used for data structures, vector store integration, embedding model wrappers, LLM wrappers, indexing, query engine).
* **Vector Database:** **ChromaDB** (Used via chromadb library and llama-index-vector-stores-chroma integration, configured for local persistence).
* **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 (Used via sentence-transformers library and llama-index-embeddings-huggingface integration). Loaded onto MPS device if available.
* **LLM:** **Mistral 7B** (Specific quantization not defined, assumed default mistral model pulled via Ollama).
* **LLM Runner:** **Ollama** (Used via llama-index-llms-ollama integration).
* **Readwise API Interaction:** requests library.
* **Environment Management:** venv.
* **Configuration:** python-dotenv loading .env file.

⠀5. Data Model & Flow
**5.1 Data Ingestion Flow (**python -m src.data_ingestion.ingest**)**
1 ingest.py main block starts.
2 Calls readwise_client.get_readwise_token() to load API key from .env.
3 Calls readwise_client.fetch_all_highlights() which loops through Readwise /export endpoint using pagination, returning a list of source dictionaries (each containing nested highlights).
4 ingest.py calls process_highlights_to_documents() with the raw data.
5 process_highlights_to_documents iterates through sources, then highlights. For each highlight:
	* Combines highlight text and note.
	* Extracts metadata from both source and highlight levels.
	* Converts tag lists (book/highlight) into comma-separated strings.
	* Creates a LlamaIndex Document object with text, metadata, and unique ID (readwise_highlight_{id}).
	* Returns a list of Document objects.
6 ingest.py main block calls store_manager.get_embedding_model() to initialize the HuggingFace embedder.
7 Calls store_manager.get_vector_store() to initialize ChromaDB persistent client and get/create the collection (readwise_highlights), returning a ChromaVectorStore.
8 Calls store_manager.add_documents_to_store(), passing the Document list, vector store, and embedding model.
9 add_documents_to_store sets Settings.embed_model, creates StorageContext, and calls VectorStoreIndex.from_documents(). This generates embeddings for all documents and upserts them into the ChromaDB collection via the ChromaVectorStore.

⠀**5.2 Query Flow (**python -m src.main**)**
1 main.py starts the run_cli() loop.
2 User enters query text.
3 main.py calls pipeline.query_knowledge_base() with the query text.
4 query_knowledge_base calls pipeline.setup_pipeline() (if not already run).
5 setup_pipeline:
	* Initializes embedding model (store_manager.get_embedding_model) and sets Settings.embed_model.
	* Initializes LLM (generator.get_llm) and sets Settings.llm.
	* Initializes vector store (store_manager.get_vector_store).
	* Loads index from store: index = VectorStoreIndex.from_vector_store(vector_store).
	* Creates query engine: query_engine = index.as_query_engine(similarity_top_k=5). (Current setting k=5).
	* Stores query_engine globally.
6 query_knowledge_base calls query_engine.query(query_text).
7 The query engine (internally):
	* Embeds the query_text using Settings.embed_model.
	* Performs similarity search in the ChromaVectorStore for the top k nodes.
	* Retrieves the text content and metadata of the top k nodes.
	* Formats a prompt containing the original query and the retrieved context.
	* Sends the prompt to the LLM specified in Settings.llm (Ollama/Mistral).
	* Receives the response text from the LLM.
	* Returns a Response object containing the response text and source nodes.
8 query_knowledge_base logs retrieved source nodes and extracts/returns the response text.
9 main.py prints the response text to the console.

⠀**5.3 Data Structures**
* LlamaIndex Document: Used to represent each highlight. Contains text (highlight + note) and metadata (dictionary of primitive types: strings, numbers, bools, None - includes highlight ID, source info, dates, comma-separated tags). id_ is set to readwise_highlight_{id}.
* **ChromaDB Collection:** Stores embeddings, document text, and metadata. Schema implicitly defined by LlamaIndex interaction.

⠀6. Implementation Details
* **Imports:** Primarily use relative imports within the src package (e.g., from .readwise_client import ..., from ..vector_store import ...). Assumes execution via python -m src.xxx.
* **Metadata Handling:** Tag lists are converted to comma-separated strings to comply with ChromaDB metadata restrictions.
* **Error Handling:** Basic try...except blocks used in main execution flows. Logging used extensively. Rate limit (429) handling added to API client.
* **Configuration:** API Key loaded from .env via python-dotenv. Model names, paths, similarity_top_k are currently hardcoded as constants or defaults but could be moved to a config file or environment variables if needed (Ref PRD FR4.2).
* **Embedding Device:** Attempts to use MPS on Apple Silicon, falls back to CPU.
* **Query Engine:** Default LlamaIndex query engine used with similarity_top_k=5. No advanced retrieval strategies (metadata filtering, MMR, etc.) implemented yet.

⠀7. Resource Management Considerations (8GB RAM)
* **LLM:** Ollama manages the Mistral 7B model (quantized). Resource usage depends on Ollama's implementation.
* **Embeddings:** all-MiniLM-L6-v2 is relatively small and efficient. Embedding generation during ingestion is a one-time cost (per document). Loading the model for querying uses RAM. MPS helps if available.
* **Vector DB:** ChromaDB persistent storage keeps the main data on disk (SSD). RAM usage during querying depends on ChromaDB's caching and query complexity but is generally manageable for this scale.
* **LlamaIndex:** Memory usage depends on the index structure loaded and query engine complexity. Loading the index from a persistent store is generally memory-efficient.

⠀8. Testing Strategy
* **Manual Testing:** Primary method used so far via direct script execution (if __name__ == "__main__":) and the CLI (main.py). Tested connection, ingestion, processing, storage, LLM response, RAG query flow.
* **Unit/Integration Tests:** Not implemented. Would require mocking API calls, DB interactions, LLM responses using pytest and unittest.mock. (Ref TDD v0.1).

⠀9. Deployment & Operations
* **Setup:** Requires Python 3, venv, pip install -r requirements.txt, Ollama installation + ollama pull mistral, Readwise API key in .env.
* **Running Ingestion:** python -m src.data_ingestion.ingest (run from project root).
* **Running Query CLI:** python -m src.main (run from project root).
* **Updates:** Manual re-run of ingestion script needed to add new highlights (no automated sync yet - Ref PRD FR1.5).

⠀10. Future Work / Scalability
* Implement automated fetching of new highlights (PRD FR1.5).
* Implement metadata filtering during retrieval (PRD FR2.4).
* Explore different embedding models or fine-tuning (PRD 5).
* Explore different chunk




#projectNeocorte