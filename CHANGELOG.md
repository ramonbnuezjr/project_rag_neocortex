# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Created initial project structure (`src`, `scripts`, `notebooks`, subdirs).
- Created initial documentation (`README.md`, `CHANGELOG.md`, `PRD.md`, `TDD.md`).
- Implemented `readwise_client.py` to fetch data from Readwise API export endpoint.
- Added pagination handling to `fetch_all_highlights` in `readwise_client.py`.
- Added basic rate limit handling (HTTP 429) to `fetch_all_highlights`.
- Created `src/data_ingestion/ingest.py` script to process raw Readwise export data.
- Implemented logic in `ingest.py` to transform nested highlight data into LlamaIndex `Document` objects with associated metadata (tags converted to strings).
- Created `src/vector_store/store_manager.py` to manage ChromaDB vector store.
- Implemented ChromaDB initialization (persistent storage) and embedding model loading (`all-MiniLM-L6-v2`) in `store_manager.py`.
- Implemented `add_documents_to_store` using `VectorStoreIndex.from_documents` to handle embedding and storage.
- Integrated `store_manager` into `ingest.py` to complete the ingestion pipeline (fetch -> process -> store).
- Created `src/llm_interface/generator.py` to manage connection to local Ollama LLM (Mistral).
- Implemented `get_llm` function using `llama_index.llms.ollama`.
- Successfully tested LLM connection and response generation.
- Created `src/rag_pipeline/pipeline.py` to manage RAG query execution.
- Integrated vector store loading, LLM initialization, and query engine setup in `pipeline.py`.
- Added logging for retrieved source nodes in `pipeline.py` for debugging.
- Successfully executed end-to-end RAG query, retrieving relevant documents and generating context-aware answers.
- Created `src/main.py` to provide an interactive command-line interface (CLI) for querying the RAG system.

### Fixed
- Resolved various `ImportError` issues related to relative imports and missing packages by using `python -m` execution and installing necessary dependencies (`llama-index-vector-stores-chroma`, etc.).
- Corrected `ValueError` during document storage by converting metadata tag lists to comma-separated strings in `ingest.py`.
- Corrected `AttributeError` in `store_manager.py` by removing direct access attempt to underlying ChromaDB collection count after indexing.
- Re-indexed ChromaDB vector store (`./chroma_db_store`) to ensure clean data state, resolving potential inconsistencies and confirming document count.
- Improved retrieval relevance for some queries by increasing `similarity_top_k` to 5 in `pipeline.py`.

### Changed
- Renamed project folder from `project_neocortex` to `project_rag_neocortex` and moved into `ai_ml` category.
- Updated `similarity_top_k` default to 5 in `pipeline.py`.

### Removed
- Redundant `ls` alias definition in `.zshrc` example.
- Problematic logging line attempting `vector_store.collection.count()` in `store_manager.py`.

