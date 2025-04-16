import logging
from llama_index.llms.ollama import Ollama
# Optional: Import Settings if you want to set the LLM globally here,
# but returning the instance is often more flexible.
# from llama_index.core.settings import Settings

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Default model to use with Ollama (ensure you have pulled this model: ollama pull mistral)
DEFAULT_OLLAMA_MODEL = "mistral"
# Default Ollama server URL
DEFAULT_OLLAMA_URL = "http://localhost:11434"
# Request timeout in seconds
DEFAULT_REQUEST_TIMEOUT = 120.0

def get_llm(model: str = DEFAULT_OLLAMA_MODEL,
            base_url: str = DEFAULT_OLLAMA_URL,
            request_timeout: float = DEFAULT_REQUEST_TIMEOUT) -> Ollama:
    """
    Initializes and returns the Ollama LLM client configured via LlamaIndex.

    Args:
        model: The name of the Ollama model to use (e.g., 'mistral').
        base_url: The base URL of the running Ollama instance.
        request_timeout: The timeout for requests to the Ollama server.

    Returns:
        An instance of the LlamaIndex Ollama LLM wrapper.
    """
    logging.info(f"Initializing Ollama LLM with model: {model}, URL: {base_url}")
    try:
        llm = Ollama(
            model=model,
            base_url=base_url,
            request_timeout=request_timeout
            # Add other parameters like temperature if needed, e.g., temperature=0.7
        )
        # Optional: Set globally if desired, but returning is more flexible
        # Settings.llm = llm
        logging.info("Ollama LLM client initialized successfully.")
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize Ollama LLM: {e}", exc_info=True)
        raise # Re-raise the exception

# --- Main execution block (for testing this module directly) ---
if __name__ == "__main__":
    logging.info("Running llm_interface.generator.py directly for testing...")

    # Ensure Ollama server is running and the model (e.g., mistral) is pulled/available
    # Example: ollama run mistral (in a separate terminal)

    try:
        # Get the LLM instance
        llm_instance = get_llm()

        # Test with a simple prompt
        test_prompt = "Explain the concept of Retrieval-Augmented Generation (RAG) in one sentence."
        logging.info(f"Sending test prompt to Ollama: '{test_prompt}'")

        # Use the .complete() method for a single response
        response = llm_instance.complete(test_prompt)

        logging.info("Received response from Ollama:")
        # The response object might have specific attributes, often .text or just str(response)
        print("-" * 20)
        print(str(response))
        print("-" * 20)
        logging.info("Direct script test completed successfully.")

    except Exception as e:
        logging.error(f"Direct script test failed. Is the Ollama server running with model '{DEFAULT_OLLAMA_MODEL}'?")
        logging.error(f"Error details: {e}", exc_info=True)

