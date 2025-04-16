import logging
import chromadb
import os
from typing import List

# LlamaIndex components
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings # Use newer Settings

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Define path for persistent ChromaDB storage relative to project root
DEFAULT_PERSIST_PATH = "./chroma_db_store"
DEFAULT_COLLECTION_NAME = "readwise_highlights"
# Use the SentenceTransformer model specified in TDD
DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_model(model_name: str = DEFAULT_EMBED_MODEL_NAME) -> HuggingFaceEmbedding:
    """Initializes and returns the HuggingFace embedding model."""
    logging.info(f"Initializing embedding model: {model_name}")
    try:
        # Try initializing with MPS device for potential speedup on M1/M2 Macs
        embed_model = HuggingFaceEmbedding(model_name=model_name, device='mps')
        logging.info("Embedding model loaded on MPS device.")
    except Exception as e:
        logging.warning(f"Failed to initialize embedding model on MPS device ({e}). Falling back to CPU.")
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        logging.info("Embedding model loaded on CPU.")
    return embed_model

def get_vector_store(persist_path: str = DEFAULT_PERSIST_PATH,
                     collection_name: str = DEFAULT_COLLECTION_NAME) -> ChromaVectorStore:
    """Initializes ChromaDB client, gets/creates a collection, and returns a LlamaIndex ChromaVectorStore."""
    logging.info(f"Initializing ChromaDB client at path: {persist_path}")
    if not os.path.exists(persist_path):
        logging.info(f"Persistent path {persist_path} does not exist. Creating directory.")
        os.makedirs(persist_path)

    db_client = chromadb.PersistentClient(path=persist_path)
    logging.info(f"Getting or creating ChromaDB collection: {collection_name}")
    chroma_collection = db_client.get_or_create_collection(collection_name)
    logging.info(f"Collection '{collection_name}' ready with {chroma_collection.count()} documents (from initial load).")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    logging.info("ChromaVectorStore initialized.")
    return vector_store

def add_documents_to_store(documents: List[Document],
                           vector_store: ChromaVectorStore,
                           embed_model: HuggingFaceEmbedding):
    """
    Adds LlamaIndex Document objects to the specified ChromaVectorStore,
    handling embedding generation via VectorStoreIndex.
    """
    if not documents:
        logging.warning("No documents provided to add to the vector store.")
        return

    logging.info(f"Adding {len(documents)} documents to the vector store...")

    Settings.embed_model = embed_model
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        logging.info("Creating index and adding documents (this may take time)...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True # Requires 'pip install tqdm'
        )
        logging.info(f"Successfully added/updated {len(documents)} documents in the index.")
        # **FIX:** Removed the problematic line below as vector_store doesn't have a public '.collection' attribute
        # logging.info(f"Collection now contains {vector_store.collection.count()} documents.")
        # If needed later, query the count via chroma_collection obtained in get_vector_store
        return index
    except Exception as e:
        # Log the specific error during index creation/document addition
        logging.error(f"Failed to add documents to vector store during index operation: {e}", exc_info=True)
        raise # Re-raise the exception after logging

# --- Main execution block (for testing this module directly) ---
# (Test block remains the same as before)
if __name__ == "__main__":
    logging.info("Running vector_store_manager.py directly for testing...")
    test_docs = [
        Document(text="This is the first test document.", metadata={"source": "test"}),
        Document(text="This is the second test document about apples.", metadata={"source": "test"}),
        Document(text="A third document, focusing on oranges.", metadata={"source": "test"}),
    ]
    try:
        test_embed_model = get_embedding_model()
        test_vector_store = get_vector_store(persist_path="./chroma_db_test", collection_name="test_collection")
        add_documents_to_store(test_docs, test_vector_store, test_embed_model)
        logging.info("Direct script test completed successfully.")
        # Optional: Clean up test database directory
        # import shutil
        # logging.info("Cleaning up test database directory: ./chroma_db_test")
        # shutil.rmtree("./chroma_db_test")
    except Exception as e:
        logging.error(f"Direct script test failed: {e}", exc_info=True)

