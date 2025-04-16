import logging
from typing import List, Dict, Any
import sys # Import sys for potential path adjustments if needed, though -m should handle it
import os

# Ensure the project root is discoverable if running with 'python -m' from root
# This might not be strictly necessary if package structure is standard

# Import from sibling directory 'readwise_client' within the same package 'src.data_ingestion'
from .readwise_client import fetch_all_highlights, get_readwise_token

# Import from sibling directory 'vector_store' within the parent package 'src'
# Requires src/__init__.py and src/vector_store/__init__.py to exist
from ..vector_store.store_manager import get_embedding_model, get_vector_store, add_documents_to_store

# Import the Document class from LlamaIndex
from llama_index.core.schema import Document

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_highlights_to_documents(raw_export_data: List[Dict[str, Any]]) -> List[Document]:
    """
    Processes the raw data fetched from Readwise API /export endpoint
    and converts highlights into LlamaIndex Document objects.

    Args:
        raw_export_data: A list of dictionaries, where each dictionary represents
                         a source (book, article) containing its highlights.

    Returns:
        A list of LlamaIndex Document objects, each representing a single highlight.
    """
    llama_documents = []
    logging.info(f"Processing {len(raw_export_data)} sources from Readwise export...")
    processed_highlight_ids = set() # Keep track of processed highlight IDs

    for source_data in raw_export_data:
        # Extract book/article level metadata
        book_metadata_keys = ['user_book_id', 'title', 'author', 'readable_title', 'source',
                              'cover_image_url', 'unique_url', 'category', 'document_note']
        book_tags_raw = source_data.get('book_tags', [])
        common_metadata = {key: source_data.get(key) for key in book_metadata_keys if source_data.get(key) is not None}
        # **FIX:** Convert book_tags list to a comma-separated string
        book_tag_names = [tag['name'] for tag in book_tags_raw]
        common_metadata['book_tags'] = ", ".join(book_tag_names) if book_tag_names else "" # Use empty string if no tags

        highlights = source_data.get('highlights', [])
        logging.debug(f"Processing {len(highlights)} highlights for source: {source_data.get('title')}")

        for highlight in highlights:
            highlight_id = highlight.get('id')
            if not highlight_id:
                logging.warning(f"Skipping highlight with missing ID in source '{source_data.get('title')}'. Data: {highlight}")
                continue

            if highlight_id in processed_highlight_ids:
                logging.debug(f"Skipping duplicate highlight ID: {highlight_id}")
                continue

            highlight_text = highlight.get('text', '')
            if not highlight_text.strip():
                logging.debug(f"Skipping highlight ID {highlight_id} due to empty text.")
                processed_highlight_ids.add(highlight_id)
                continue

            highlight_note = highlight.get('note', '')
            document_text = highlight_text
            if highlight_note:
                document_text += f"\n\nNote: {highlight_note}"

            highlight_metadata = common_metadata.copy()
            highlight_tags_raw = highlight.get('tags', [])
            # **FIX:** Convert highlight_tags list to a comma-separated string
            highlight_tag_names = [tag['name'] for tag in highlight_tags_raw]
            highlight_metadata.update({
                'highlight_id': highlight_id,
                'highlighted_at': highlight.get('highlighted_at'),
                'highlight_url': highlight.get('url'),
                'updated_at': highlight.get('updated_at'),
                'color': highlight.get('color'),
                'highlight_tags': ", ".join(highlight_tag_names) if highlight_tag_names else "" # Use empty string if no tags
            })
            # Clean up None values from metadata AFTER potential empty strings are added
            highlight_metadata = {k: v for k, v in highlight_metadata.items() if v is not None}


            try:
                document = Document(
                    text=document_text,
                    metadata=highlight_metadata,
                    id_=f"readwise_highlight_{highlight_id}"
                )
                llama_documents.append(document)
                processed_highlight_ids.add(highlight_id) # Mark as processed
            except Exception as e:
                logging.error(f"Error creating Document for highlight ID {highlight_id}: {e}")
                logging.error(f"Highlight data: {highlight}")
                logging.error(f"Metadata attempted: {highlight_metadata}")

    logging.info(f"Finished processing. Created {len(llama_documents)} unique LlamaIndex Document objects.")
    return llama_documents

# --- Main execution block ---
if __name__ == "__main__":
    logging.info("Starting Readwise data ingestion, processing, and storage...")
    try:
        # 1. Fetch raw data using the client
        logging.info("Step 1: Fetching data from Readwise API...")
        api_key = get_readwise_token()
        raw_data = fetch_all_highlights(api_key)

        if raw_data:
            # 2. Process raw data into LlamaIndex Documents
            logging.info("Step 2: Processing fetched data into LlamaIndex Documents...")
            processed_documents = process_highlights_to_documents(raw_data)

            # 3. Store processed documents in ChromaDB
            if processed_documents:
                logging.info("Step 3: Initializing components and storing documents in vector store...")
                # Initialize embedding model and vector store using the manager
                embed_model = get_embedding_model()
                # Use default path and collection name from store_manager
                vector_store = get_vector_store()

                # Add the processed documents to the store
                # This step handles embedding generation and storage
                add_documents_to_store(processed_documents, vector_store, embed_model)

                logging.info("Ingestion, processing, and storage pipeline completed successfully.")
            else:
                logging.info("No documents were processed, skipping storage.")
        else:
            logging.info("No raw data fetched from Readwise, pipeline finished.")

    except ValueError as e:
        logging.info(f"Exiting due to configuration error: {e}")
    except Exception as e:
        logging.error(f"Ingestion pipeline failed with an unexpected error: {e}", exc_info=True)

