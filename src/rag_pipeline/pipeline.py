import logging
import sys
import os

# LlamaIndex core components
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.schema import NodeWithScore # Import for type hinting if needed

# Import functions from our other modules using relative paths
try:
    from ..llm_interface.generator import get_llm
    from ..vector_store.store_manager import get_vector_store, get_embedding_model
except ImportError:
    logging.error("Failed relative imports. Ensure script is run as module from project root "
                  "and __init__.py files exist.")
    sys.exit("Import errors, check project structure and execution method.")


# Configure basic logging
# Change level to DEBUG to see more LlamaIndex internal logs if needed,
# but INFO should be sufficient for our custom logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global variable to hold the initialized query engine ---
_query_engine_global: BaseQueryEngine | None = None

def setup_pipeline(similarity_top_k: int = 5) -> BaseQueryEngine:
    """
    Sets up the RAG query pipeline by initializing components and loading the index.

    Args:
        similarity_top_k: The number of top similar documents to retrieve from the vector store.

    Returns:
        An initialized LlamaIndex query engine.
    """
    global _query_engine_global
    if _query_engine_global is not None:
        logging.debug("Query engine already initialized.")
        return _query_engine_global

    logging.info("Setting up RAG query pipeline...")

    # 1. Configure embedding model and LLM via Settings
    logging.info("Initializing embedding model...")
    Settings.embed_model = get_embedding_model()
    logging.info("Initializing LLM...")
    Settings.llm = get_llm()

    # 2. Load the vector store
    logging.info("Loading vector store...")
    vector_store = get_vector_store()

    # 3. Load the index from the vector store
    logging.info("Loading index from vector store...")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    logging.info("Index loaded successfully.")

    # 4. Create the query engine
    logging.info(f"Creating query engine with similarity_top_k={similarity_top_k}...")
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
    logging.info("Query engine created successfully.")

    _query_engine_global = query_engine
    logging.info("RAG pipeline setup complete.")
    return _query_engine_global

def query_knowledge_base(query_text: str) -> str:
    """
    Queries the knowledge base using the initialized RAG pipeline. Logs retrieved source nodes.

    Args:
        query_text: The user's question.

    Returns:
        The response string from the LLM.
    """
    logging.info(f"Received query: '{query_text}'")
    try:
        query_engine = setup_pipeline()
        logging.info("Querying the engine...")
        response = query_engine.query(query_text)
        logging.info("Received response object from query engine.")

        # **MODIFICATION START: Log source nodes**
        source_nodes = getattr(response, 'source_nodes', [])
        if source_nodes:
            logging.info(f"Retrieved {len(source_nodes)} source nodes:")
            for i, node_with_score in enumerate(source_nodes):
                node = node_with_score.node
                score = node_with_score.score
                logging.info(f"--- Source Node {i+1} (Score: {score:.4f}) ---")
                logging.info(f"Node ID: {node.node_id}")
                # Log metadata, which contains useful context like source title, author etc.
                logging.info(f"Metadata: {node.metadata}")
                # Log a slightly longer snippet of the text
                logging.info(f"Text: {node.text[:250]}...")
                logging.info("-" * (18 + len(str(i+1)))) # Divider length matches header
        else:
            # This case might happen if retrieval fails or returns nothing
            logging.warning("Query execution did not return any source nodes.")
        # **MODIFICATION END**

        response_text = str(getattr(response, 'response', str(response))).strip()
        logging.debug(f"Raw response object type: {type(response)}")
        logging.info(f"Extracted response text: {response_text}") # Log the final response text

        return response_text

    except Exception as e:
        logging.error(f"Error during query execution: {e}", exc_info=True)
        return "Sorry, I encountered an error while processing your query."

# --- Main execution block (for testing this module directly) ---
if __name__ == "__main__":
    logging.info("Running rag_pipeline.pipeline.py directly for testing...")
    # Test query likely to have results in the sample data
    test_query = "What is mentioned about Ryan Holiday?"
    # Alternative test query:
    # test_query = "What is RAG?"

    try:
        final_response = query_knowledge_base(test_query)
        print("\n" + "="*30 + " Query Result " + "="*30)
        print(f"Query: {test_query}")
        print(f"Response:\n{final_response}")
        print("="*74)

    except Exception as e:
        logging.error(f"Direct script test failed: {e}", exc_info=True)

