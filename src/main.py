import logging
import sys
import os

# Configure basic logging (optional, could rely on pipeline's logging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Import the core query function ---
# Use a try-except block for robustness, especially if running in different ways
try:
    # Assuming main.py is in src/ and pipeline.py is in src/rag_pipeline/
    # This relative import works when running as 'python -m src.main'
    from .rag_pipeline.pipeline import query_knowledge_base
except ImportError:
    logging.error("Failed relative import. Attempting import assuming 'src' is discoverable.")
    # Fallback if run differently or project structure changes
    # This requires the project root directory to be in Python's path
    try:
        from rag_pipeline.pipeline import query_knowledge_base
    except ImportError:
        logging.error("Could not import query_knowledge_base. "
                      "Ensure __init__.py files exist and script is run correctly "
                      "(e.g., 'python -m src.main' from project root).")
        sys.exit("Import setup failed.")


def run_cli():
    """Runs the command-line interface loop."""
    print("\n=============================================")
    print(" Welcome to Project Neocortex RAG System")
    print("=============================================")
    print("Ask questions based on your Readwise highlights.")
    print("Type 'quit' or 'exit' to stop.")
    print("---------------------------------------------")

    while True:
        try:
            # Get user input
            query_text = input("Your query: ").strip()

            # Check for exit command
            if query_text.lower() in ['quit', 'exit']:
                print("Exiting RAG system. Goodbye!")
                break

            # Ensure query is not empty
            if not query_text:
                continue

            # Call the RAG pipeline to get the response
            print("Processing your query...")
            response = query_knowledge_base(query_text)

            # Print the response
            print("\nResponse:")
            print("-" * 10)
            print(response)
            print("-" * 10 + "\n")

        except EOFError:
            # Handle Ctrl+D or end-of-input gracefully
            print("\nExiting RAG system. Goodbye!")
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting RAG system. Goodbye!")
            break
        except Exception as e:
            # Catch other potential errors during query execution
            logging.error(f"An error occurred in the CLI loop: {e}", exc_info=True)
            print("An unexpected error occurred. Please try again or check logs.")


# --- Main execution block ---
if __name__ == "__main__":
    run_cli()

