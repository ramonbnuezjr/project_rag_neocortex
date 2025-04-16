import os
import requests
from dotenv import load_dotenv
import logging
import time # Import time for potential rate limit handling

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
# This looks for a .env file in the current directory or parent directories
# Ensure this script is run in a context where it can find the .env file (e.g., from project root)
load_dotenv()

# --- Constants ---
READWISE_API_BASE_URL = "https://readwise.io/api/v2/"

def get_readwise_token():
    """Retrieves the Readwise API token from environment variables."""
    token = os.getenv("READWISE_API_KEY")
    if not token:
        logging.error("READWISE_API_KEY not found in environment variables. Make sure it's set in your .env file.")
        raise ValueError("Missing Readwise API Key")
    # Log only the first few characters for verification, never the full key
    logging.debug(f"Readwise API Key loaded (starts with: {token[:4]}...)")
    return token

def fetch_all_highlights(api_token: str):
    """
    Fetches all data from the Readwise /export endpoint, handling pagination.
    """
    full_data = []
    next_page_cursor = None
    api_url = f"{READWISE_API_BASE_URL}export/"
    headers = {"Authorization": f"Token {api_token}"}
    logging.info("Starting fetch for all Readwise highlights...")

    while True:
        params = {}
        current_url = api_url # Define url for logging
        if next_page_cursor:
            params['pageCursor'] = next_page_cursor
            # Log only the cursor, not the full URL with potentially sensitive token in headers
            logging.info(f"Fetching next page with cursor: {next_page_cursor}")
        else:
            logging.info(f"Fetching first page from base export URL.")

        try:
            response = requests.get(url=api_url, headers=headers, params=params)

            # Handle potential rate limiting (429 Too Many Requests)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60)) # Default to 60s wait
                logging.warning(f"Rate limit hit. Waiting for {retry_after} seconds.")
                time.sleep(retry_after)
                continue # Retry the same request

            # Check for other errors
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            # Process successful response
            data = response.json()
            results = data.get('results', [])
            if results:
                full_data.extend(results)
                logging.info(f"Fetched {len(results)} source items. Total fetched so far: {len(full_data)}")
            else:
                # This might happen on the very last page or if the export is empty
                logging.info("No results found on this page.")

            # Check for next page cursor
            next_page_cursor = data.get('nextPageCursor')
            if not next_page_cursor:
                logging.info("No more pages to fetch (nextPageCursor is null or missing).")
                break # Exit loop if no next page

        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred during the API request: {e}")
            # Depending on the error, you might want to retry or break
            break # Simple break for now
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            break # Simple break for now

    logging.info(f"Finished fetching. Total source items retrieved: {len(full_data)}")
    return full_data

# --- Test function (Optional - can be removed or kept for direct testing) ---
def test_readwise_connection(api_token: str):
    """
    Tests the connection to the Readwise API by attempting to list export formats.
    Uses the /export endpoint as a simple authenticated check.
    (This was the original test function)
    """
    api_url = f"{READWISE_API_BASE_URL}export/"
    headers = {"Authorization": f"Token {api_token}"}
    logging.info(f"Attempting to connect to Readwise API at: {api_url}")

    try:
        # Make a request for only the first page (no params)
        response = requests.get(url=api_url, headers=headers)
        response.raise_for_status() # Check for HTTP errors

        logging.info("Successfully connected to Readwise API (test_readwise_connection)!")
        response_data = response.json()
        logging.info(f"Test connection response keys: {list(response_data.keys())}")
        return True

    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred during the API request test: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during test: {e}")
        return False

# --- Main execution block (for testing this script directly) ---
if __name__ == "__main__":
    logging.info("Running readwise_client.py directly for testing...")
    try:
        api_key = get_readwise_token()
        # Perform the connection test first
        if test_readwise_connection(api_key):
             # If test passes, try fetching all highlights
             logging.info("Connection test passed. Now attempting to fetch all highlights...")
             all_highlights_data = fetch_all_highlights(api_key)
             logging.info(f"Direct script run: Successfully fetched {len(all_highlights_data)} total source items.")
        else:
            logging.warning("Connection test failed. Skipping full fetch.")

    except ValueError as e:
        logging.info(f"Exiting due to configuration error: {e}")
    except Exception as e:
        logging.error(f"Direct script run failed with an unexpected error: {e}", exc_info=True)


