import os
import json
import hashlib
import logging
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup  # type: ignore[import]
from typing import Set, List, Dict, Any, Union
from eidosian_core import eidosian

# Setup detailed logging configuration for robust debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tika_searxng_scraper")

# Default configuration for endpoints and storage
DEFAULT_SEARXNG_URL: str = "http://192.168.4.73:8888"
DEFAULT_TIKA_URL: str = "http://192.168.4.73:9998"
DEFAULT_DOCUMENT_STORE: str = "./document_store"
DEFAULT_PROCESSED_LOG: str = "processed_urls.json"

# Ensure document store exists
os.makedirs(DEFAULT_DOCUMENT_STORE, exist_ok=True)


# ---------------------------------------------------------------------------
# Processed URL Store
# Handles tracking of which URLs have been processed to avoid duplicates.
class ProcessedUrlStore:
    """
    Manages the storage and retrieval of processed URLs to prevent duplicate processing.
    """

    def __init__(self, log_file: str = DEFAULT_PROCESSED_LOG):
        self.log_file = log_file
        self.processed_urls: Set[str] = set()
        self._load()

    def _load(self) -> None:
        """Loads processed URLs from the log file."""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, "r") as f:
                    self.processed_urls = set(json.load(f))
                logger.info("Loaded processed URLs from log.")
            else:
                self.processed_urls = set()
                logger.info("Processed URLs log not found. Starting with an empty set.")
        except Exception as e:
            logger.error(f"Failed to load processed URLs from {self.log_file}: {e}")
            self.processed_urls = set()

    @eidosian()
    def is_processed(self, url: str) -> bool:
        """Returns True if the URL has already been processed."""
        return url in self.processed_urls

    @eidosian()
    def mark_as_processed(self, url: str) -> None:
        """
        Marks a URL as processed and updates the log file using an atomic write strategy.
        """
        self.processed_urls.add(url)
        try:
            temp_log = f"{self.log_file}.tmp"
            with open(temp_log, "w") as f:
                json.dump(list(self.processed_urls), f, indent=2)
            os.replace(temp_log, self.log_file)
            logger.info(f"Processed URL saved: {url}")
        except Exception as e:
            logger.error(f"Error saving processed URL {url}: {e}")


# ---------------------------------------------------------------------------
# Tika Client
# Responsible for extracting text and metadata from a URL using Tika.
class TikaClient:
    """
    Client for interacting with Tika's /rmeta endpoint to extract text and metadata.
    """

    def __init__(self, tika_url: str = DEFAULT_TIKA_URL):
        self.tika_url = tika_url

    @eidosian()
    def extract(self, url: str) -> List[Dict[str, Any]]:
        """
        Fetches content from a URL and extracts text and metadata using Tika.

        Args:
            url: The URL from which to fetch and extract content.

        Returns:
            A list of dictionaries (one per document part, including attachments)
            containing extracted text and related metadata.
        """
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            content_type = response.headers.get(
                "Content-Type", "application/octet-stream"
            )
            logger.info(
                f"Fetching content from {url} with Content-Type: {content_type}"
            )

            headers = {"Content-type": content_type, "Accept": "application/json"}
            tika_endpoint = f"{self.tika_url}/rmeta"
            tika_response = requests.put(
                tika_endpoint, headers=headers, data=response.content, timeout=20
            )
            tika_response.raise_for_status()
            logger.info(f"Tika extraction successful for URL: {url}")
            return json.loads(tika_response.text)
        except requests.RequestException as re:
            logger.error(f"HTTP error during Tika extraction for URL {url}: {re}")
        except Exception as e:
            logger.error(
                f"Unexpected error extracting content with Tika for URL {url}: {e}"
            )
        return []


# ---------------------------------------------------------------------------
# Document Store
# Responsible for saving extracted document parts in a structured way.
class DocumentStore:
    """
    Handles saving of extracted document parts as JSON files in a structured directory.
    """

    def __init__(self, base_dir: str = DEFAULT_DOCUMENT_STORE):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    @eidosian()
    def save_document_parts(
        self, url: str, content_parts: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> None:
        """
        Saves each document/attachment part in its own JSON file within a subfolder
        named after the MD5 hash of the original URL.

        Args:
            url: The original document URL.
            content_parts: A list of dictionaries from Tika's extraction.
            metadata: Additional information (such as crawl depth) to store alongside.
        """
        try:
            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
            doc_folder = os.path.join(self.base_dir, url_hash)
            os.makedirs(doc_folder, exist_ok=True)
            logger.info(f"Saving document parts to folder: {doc_folder}")

            for i, part in enumerate(content_parts):
                filename = os.path.join(doc_folder, f"part_{i}.json")
                with open(filename, "w") as f:
                    json.dump(
                        {
                            "url": url,
                            "extraction_metadata": metadata,
                            "document_part": part,
                        },
                        f,
                        indent=2,
                    )
                logger.debug(f"Saved document part {i} for URL: {url}")
            logger.info(
                f"Successfully saved {len(content_parts)} document part(s) for {url}."
            )
        except Exception as e:
            logger.error(f"Error saving document parts for {url}: {e}")


# ---------------------------------------------------------------------------
# SearxNG Searcher
# Handles performing searches against the SearxNG instance.
class SearxngSearcher:
    """
    Performs searches against the configured SearxNG instance.
    """

    def __init__(self, searxng_url: str = DEFAULT_SEARXNG_URL):
        self.searxng_url = searxng_url

    @eidosian()
    def search(self, query: str, num_results: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Performs a SearxNG search.

        Args:
            query: The search query string.
            num_results: The number of search results to retrieve.
            **kwargs: Additional search parameters.

        Returns:
            A dictionary containing the JSON response from SearxNG.
        """
        params: Dict[str, Union[str, int]] = {
            "q": query,
            "format": "json",
            "engines": "google,bing,academic",
            "categories": "science",
            "safesearch": 1,
            "lang": "en",
            "count": num_results,
        }
        params.update(kwargs)
        try:
            response = requests.get(
                urljoin(self.searxng_url, "/search"), params=params, timeout=10
            )
            response.raise_for_status()
            logger.info(
                f"Search successful for query: '{query}' with {num_results} results."
            )
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Search request failed for query '{query}': {e}")
            return {}


# ---------------------------------------------------------------------------
# Link Extractor
# Provides utility functions for extracting internal links from HTML content.
class LinkExtractor:
    """
    Utility class for extracting internal links from HTML content relative
    to a given base URL.
    """

    @staticmethod
    def extract_links(base_url: str, html: bytes) -> List[str]:
        base_domain = urlparse(base_url).netloc
        soup = BeautifulSoup(html, "html.parser")
        links: List[str] = []
        for tag in soup.find_all("a", href=True):
            absolute_url = urljoin(base_url, tag["href"])
            if urlparse(absolute_url).netloc == base_domain:
                links.append(absolute_url)
        return links


# ---------------------------------------------------------------------------
# Crawler
# Responsible for recursively crawling URLs, extracting content, and saving results.
class Crawler:
    """
    Recursively crawls links within the same domain, extracts content using Tika,
    and saves document parts via the DocumentStore.
    """

    def __init__(
        self,
        tika_client: TikaClient,
        document_store: DocumentStore,
        processed_store: ProcessedUrlStore,
    ):
        self.tika_client = tika_client
        self.document_store = document_store
        self.processed_store = processed_store

    def _fetch_content(
        self, url: str, timeout: int = 10
    ) -> Union[requests.Response, None]:
        """
        Helper method to fetch content from a URL.
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"HTTP error fetching {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
        return None

    @eidosian()
    def crawl_links(self, start_url: str, depth: int = 2) -> None:
        """
        Crawls a given URL and its internal links up to a specified depth.

        Args:
            start_url: The URL to begin crawling.
            depth: Maximum recursion depth; a depth of 0 stops further recursion.
        """
        if self.processed_store.is_processed(start_url):
            logger.info(f"Skipping already processed URL: {start_url}")
            return

        logger.info(f"Crawling URL: {start_url} with depth: {depth}")
        response = self._fetch_content(start_url)
        if not response:
            return

        try:
            # Extract internal links using the LinkExtractor
            internal_links = LinkExtractor.extract_links(start_url, response.content)

            # Extract document parts using Tika
            content_parts = self.tika_client.extract(start_url)
            if content_parts:
                self.document_store.save_document_parts(
                    start_url, content_parts, {"depth": depth}
                )
                self.processed_store.mark_as_processed(start_url)
            else:
                logger.warning(f"No content parts extracted for URL: {start_url}")

            # Recursively crawl extracted links if depth permits
            if depth > 0:
                for link in internal_links:
                    self.crawl_links(link, depth=depth - 1)
        except Exception as e:
            logger.error(f"Unexpected error while crawling {start_url}: {e}")


# ---------------------------------------------------------------------------
# ScraperCrawler
# Orchestrates search and crawl operations through well-defined interfaces.
class ScraperCrawler:
    """
    Orchestrates the search and crawling operations.

    Provides a clean interface to perform a search via SearxNG, then
    crawl the resulting URLs to extract content (including attachments and images),
    which are saved and returned in a structured format.
    """

    def __init__(
        self,
        searxng_url: str = DEFAULT_SEARXNG_URL,
        tika_url: str = DEFAULT_TIKA_URL,
        document_store_dir: str = DEFAULT_DOCUMENT_STORE,
        processed_log: str = DEFAULT_PROCESSED_LOG,
    ):
        self.processed_store = ProcessedUrlStore(log_file=processed_log)
        self.tika_client = TikaClient(tika_url=tika_url)
        self.document_store = DocumentStore(base_dir=document_store_dir)
        self.searcher = SearxngSearcher(searxng_url=searxng_url)
        self.crawler = Crawler(
            self.tika_client, self.document_store, self.processed_store
        )

    @eidosian()
    def search_and_crawl(
        self, query: str, num_results: int = 10, depth: int = 2, **kwargs
    ) -> Dict[str, Any]:
        """
        Performs a SearxNG search and crawls the returned URLs to extract and save content.

        Args:
            query: The search query string.
            num_results: Number of search results to retrieve.
            depth: Crawl recursion depth for each URL.
            **kwargs: Additional parameters for the search.

        Returns:
            A cleaned dictionary of search results along with the extraction details.
        """
        search_results = self.searcher.search(query, num_results, **kwargs)
        cleaned_results: Dict[str, Any] = {"results": []}
        results = search_results.get("results", [])
        if not results:
            logger.warning(f"No search results returned for query: '{query}'")
            return cleaned_results

        for result in results:
            url = result.get("url")
            if url and not self.processed_store.is_processed(url):
                logger.info(f"Processing URL: {url}")
                self.crawler.crawl_links(url, depth=depth)
            else:
                logger.info(f"Skipping URL (already processed or missing): {url}")
            cleaned_results["results"].append(result)
        return cleaned_results


# ---------------------------------------------------------------------------
# Utility function for search and crawl operations.
@eidosian()
def search(query: str, **kwargs) -> Dict[str, Any]:
    """
    Utility function to perform a search and crawl using ScraperCrawler.
    This function requires only the query, but additional parameters
    can be provided for extensibility.

    Args:
        query: The search query string.
        **kwargs: Additional parameters for search configuration.

    Returns:
        A dictionary containing cleaned search results along with extraction details.
    """
    scraper_crawler = ScraperCrawler()
    return scraper_crawler.search_and_crawl(query, **kwargs)


# ---------------------------------------------------------------------------
# Main entry point for interactive use.
@eidosian()
def main() -> None:
    """
    Main entry point.

    Prompts for a search query, performs a SearxNG search,
    and crawls each resulting URL to extract and save content (including attachments)
    using Tika.
    """
    try:
        query = input("Enter a topic to search: ").strip()
        if not query:
            logger.error("No topic entered. Exiting.")
            return

        results = search(query)
        logger.info(f"Search and crawl completed for query: '{query}'")
        print(json.dumps(results, indent=2))
    except Exception as e:
        logger.error(f"An error occurred in main execution: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting.")
