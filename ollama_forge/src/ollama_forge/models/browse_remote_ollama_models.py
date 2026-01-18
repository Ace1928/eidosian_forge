"""
Eidosian Ollama Library Scraper and Search Tool
================================================

A surgical scraper, indexer, and search system for the Ollama model library
(https://ollama.com/library). Creates a lightning-fast local search index
with zero compromises on completeness or accuracy.

Features:
- Recursive page traversal to capture all available models
- Detailed model information extraction with depth-first discovery
- Lightning-fast local JSON indexing with minimal footprint
- Semantic search with multiple filter options and token matching
- Clean separation of concerns: scraping, indexing, searching
- Fully typed and documented for maximum IDE integration

Requirements:
- Python 3.10+
- requests
- beautifulsoup4

Authorship:
- Eidos, Lloyd Handyside
"""

import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, ResultSet, Tag

# Type definitions
T = TypeVar("T")
R = TypeVar("R")
ModelData = Dict[str, Union[str, List[str], Dict[str, str], None]]
ModelCollection = List[ModelData]
SearchResult = Tuple[int, "OllamaModel"]
SearchResultList = List[SearchResult]


# Define ModelFilter as a proper type rather than a variable
class ModelFilter(Protocol):
    """Protocol for model filter functions."""

    def __call__(self, model: "OllamaModel") -> bool: ...


# Constants
OLLAMA_LIBRARY_URL: str = "https://ollama.com/library"
DEFAULT_INDEX_PATH: str = "ollama_model_index.json"
DEFAULT_DETAILED_INDEX_PATH: str = "ollama_model_details.json"
REQUEST_TIMEOUT: int = 30
MAX_RETRIES: int = 3
RETRY_DELAY: int = 2
CONCURRENT_REQUESTS: int = 5


# Type definitions for callable objects
class Callable(Protocol):
    """Protocol for callable objects to satisfy type checking."""

    def __call__(self, *args: object, **kwargs: object) -> object: ...


@dataclass
class ScraperConfig:
    """Configuration parameters for the Ollama library scraper.

    Controls the behavior of the scraper including selectors, paths, and
    performance settings for HTTP requests and concurrency.

    Attributes:
        base_url: Base URL for the Ollama library.
        card_selector: CSS selector for model cards.
        name_selector: CSS selector for model names within cards.
        description_selector: CSS selector for model descriptions within cards.
        tag_selector: CSS selector for model tags.
        size_selector: CSS selector for model size information.
        parameter_selector: CSS selector for model parameter information.
        index_path: Path to save the basic model index.
        detailed_index_path: Path to save the detailed model information.
        timeout: Request timeout in seconds.
        retries: Maximum number of retry attempts for failed requests.
        retry_delay: Delay between retries in seconds.
        fetch_detailed_info: Whether to fetch detailed information for each model.
        concurrent_requests: Maximum number of concurrent requests for detailed fetching.
    """

    base_url: str = OLLAMA_LIBRARY_URL
    card_selector: str = "li[x-test-model]"
    name_selector: str = "h2"
    description_selector: str = "p"
    tag_selector: str = ".tag"
    size_selector: str = ".size"
    parameter_selector: str = ".parameters"
    index_path: str = DEFAULT_INDEX_PATH
    detailed_index_path: str = DEFAULT_DETAILED_INDEX_PATH
    timeout: int = REQUEST_TIMEOUT
    retries: int = MAX_RETRIES
    retry_delay: int = RETRY_DELAY
    fetch_detailed_info: bool = True
    concurrent_requests: int = CONCURRENT_REQUESTS

    def model_url(self, path: str) -> str:
        """Generate full model URL from a path.

        Ensures proper URL formation by handling path normalization.

        Args:
            path: Relative path to the model.

        Returns:
            str: Full URL to the model.
        """
        if not path:
            return self.base_url

        # Convert any path to string and normalize
        path_str = str(path).lstrip("/")

        # Use urllib's urljoin for robust URL joining
        return urljoin(self.base_url, path_str)


@dataclass
class ModelMetadata:
    """Detailed metadata for an Ollama model.

    Contains additional information that might be extracted from a model's
    dedicated page rather than the library listing.

    Attributes:
        size: Size of the model file.
        parameters: Number of parameters in the model.
        license: License information for the model.
        author: Author or organization that created the model.
        last_updated: When the model was last updated.
        download_count: Number of times the model has been downloaded.
        capabilities: List of specific capabilities the model has.
        languages: Languages supported by the model.
        context_length: Maximum context length the model supports.
        quantization: Quantization level of the model.
    """

    size: Optional[str] = None
    parameters: Optional[str] = None
    license: Optional[str] = None
    author: Optional[str] = None
    last_updated: Optional[str] = None
    download_count: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    context_length: Optional[str] = None
    quantization: Optional[str] = None


@dataclass
class OllamaModel:
    """Structured representation of an Ollama model.

    Complete representation of a model with all available information
    from both list views and detailed pages.

    Attributes:
        name: Name of the model.
        description: Description of the model.
        url: URL to the model page.
        tags: List of tags associated with the model.
        metadata: Detailed metadata about the model.
    """

    name: str
    description: str
    url: str
    tags: List[str] = field(default_factory=list)
    metadata: ModelMetadata = field(default_factory=ModelMetadata)

    def to_dict(self) -> ModelData:
        """Convert model to dictionary representation for serialization.

        Returns:
            ModelData: Dictionary representation of the model.
        """
        result = asdict(self)
        # Clean up empty values for cleaner JSON
        for key, value in list(result.items()):
            if isinstance(value, dict):
                result[key] = {
                    k: v for k, v in value.items() if v is not None and v != []
                }
        return result

    def satisfies_filter(self, filter_func: ModelFilter) -> bool:
        """Check if this model satisfies a filter function.

        Args:
            filter_func: Filter function to apply to this model.

        Returns:
            bool: True if the model satisfies the filter, False otherwise.
        """
        return filter_func(self)


class OllamaLibraryError(Exception):
    """Raised when scraping the Ollama Library fails."""

    pass


class IndexingError(Exception):
    """Raised when indexing or writing data to the local index fails."""

    pass


class HttpClient:
    """HTTP client with retry logic for robust web requests."""

    def __init__(self, config: ScraperConfig):
        """Initialize HTTP client with configuration.

        Args:
            config: Scraper configuration.
        """
        self.config = config
        self.session = requests.Session()
        # Setup a proper user agent for more reliable scraping
        self.session.headers.update(
            {
                "User-Agent": "Eidosian-Ollama-Explorer/1.0 (compatible; educational-use-only)"
            }
        )

    def get(self, url: str, params: Optional[Dict[str, str]] = None) -> str:
        """Make a GET request with retry logic.

        Args:
            url: URL to request.
            params: Optional query parameters.

        Returns:
            str: Response text content.

        Raises:
            OllamaLibraryError: If the request fails after retries.
        """
        for attempt in range(self.config.retries):
            try:
                response = self.session.get(
                    url, params=params, timeout=self.config.timeout
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                if attempt == self.config.retries - 1:
                    raise OllamaLibraryError(f"Failed to fetch {url}: {e}")
                time.sleep(self.config.retry_delay)

        # This line ensures the function always returns a string on all code paths
        # Even though this should never be reached due to the exception above
        raise OllamaLibraryError(
            f"Failed to fetch {url} after {self.config.retries} attempts"
        )


class DetailedModelExtractor:
    """Extracts detailed model information from individual model pages."""

    def __init__(self, http_client: HttpClient):
        """Initialize the detailed model extractor.

        Args:
            http_client: HTTP client for making requests.
        """
        self.http_client = http_client

    def extract_model_details(self, model: OllamaModel) -> None:
        """Extract detailed information from a model's dedicated page.

        Updates the model's metadata in-place with information scraped from
        the model's detailed page.

        Args:
            model: The model to update with detailed information.
        """
        try:
            html_content = self.http_client.get(model.url)
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract all possible metadata from the detail page
            metadata = model.metadata

            # Extract size information (if not already present)
            if not metadata.size:
                size_elem = soup.select_one(".size") or soup.select_one("[data-size]")
                if size_elem:
                    metadata.size = size_elem.text.strip()

            # Extract parameter count (if not already present)
            if not metadata.parameters:
                params_elem = soup.select_one(".parameters") or soup.select_one(
                    "[data-parameters]"
                )
                if params_elem:
                    metadata.parameters = params_elem.text.strip()

            # Extract license information
            license_elem = soup.select_one(".license") or soup.select_one(
                "[data-license]"
            )
            if license_elem:
                metadata.license = license_elem.text.strip()

            # Extract author information
            author_elem = soup.select_one(".author") or soup.select_one("[data-author]")
            if author_elem:
                metadata.author = author_elem.text.strip()

            # Extract update timestamp
            updated_elem = soup.select_one(".updated") or soup.select_one(
                "[data-updated]"
            )
            if updated_elem:
                metadata.last_updated = updated_elem.text.strip()

            # Extract download count
            downloads_elem = soup.select_one(".downloads") or soup.select_one(
                "[data-downloads]"
            )
            if downloads_elem:
                metadata.download_count = downloads_elem.text.strip()

            # Extract context length
            context_elem = soup.select_one(".context") or soup.select_one(
                "[data-context-length]"
            )
            if context_elem:
                metadata.context_length = context_elem.text.strip()

            # Extract quantization
            quant_elem = soup.select_one(".quantization") or soup.select_one(
                "[data-quantization]"
            )
            if quant_elem:
                metadata.quantization = quant_elem.text.strip()

            # Extract capabilities
            capability_elems = soup.select(".capability") or soup.select(
                "[data-capability]"
            )
            if capability_elems:
                metadata.capabilities = [elem.text.strip() for elem in capability_elems]

            # Extract supported languages
            language_elems = soup.select(".language") or soup.select("[data-language]")
            if language_elems:
                metadata.languages = [elem.text.strip() for elem in language_elems]

            # Look for any paragraphs that might contain metadata
            paragraphs = soup.select("p")
            for p in paragraphs:
                text = p.text.lower()
                # Extract context window information
                if "context" in text and "window" in text and "k" in text:
                    matches = re.search(r"(\d+)k", text)
                    if matches and not metadata.context_length:
                        metadata.context_length = matches.group(0)

                # Extract parameter count from text
                if "parameters" in text or "param" in text:
                    matches = re.search(r"(\d+(\.\d+)?)b", text.lower())
                    if matches and not metadata.parameters:
                        metadata.parameters = matches.group(0)

            # Success! Model metadata has been updated in-place
        except Exception as e:
            print(f"⚠️ Warning: Failed to extract detailed info for {model.name}: {e}")


class OllamaLibraryScraper:
    """Scraper for the Ollama Library with recursive pagination traversal.

    Implements a depth-first web crawler specialized for the Ollama model library,
    handling pagination, nested navigation, and concurrent detail extraction.
    The scraper maintains visited URL tracking to prevent redundant requests
    and provides fault-tolerant extraction with clear error boundaries.

    Attributes:
        config: Configuration parameters controlling scraper behavior
        http_client: Client for making HTTP requests with retry logic
        detail_extractor: Component for extracting detailed model information
        visited_urls: Set of URLs already visited to prevent cycles
        models: Collection of scraped model information
        semaphore: Concurrency control for parallel detail fetching
    """

    def __init__(self, config: ScraperConfig = ScraperConfig()) -> None:
        """Initialize the Ollama library scraper with configuration.

        Args:
            config: Scraper configuration parameters. Defaults to standard config.
        """
        self.config: ScraperConfig = config
        self.http_client: HttpClient = HttpClient(config)
        self.detail_extractor: DetailedModelExtractor = DetailedModelExtractor(
            self.http_client
        )
        self.visited_urls: Set[str] = set()
        self.models: List[OllamaModel] = []
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(
            config.concurrent_requests
        )

    def scrape(self) -> List[OllamaModel]:
        """Execute a complete scrape of the Ollama Library.

        Performs a two-phase extraction process:
        1. Recursively traverses the library pages to discover all models
        2. Optionally fetches detailed information for each model concurrently

        Returns:
            List[OllamaModel]: Collection of scraped model objects with metadata

        Raises:
            OllamaLibraryError: If any part of the scraping process fails
        """
        self.visited_urls.clear()
        self.models.clear()

        try:
            # First phase: discover all models through recursive page traversal
            self._scrape_page(self.config.base_url)

            # Second phase: enrich models with detailed information if enabled
            if self.config.fetch_detailed_info and self.models:
                print(
                    f"📊 Fetching detailed information for {len(self.models)} models..."
                )
                self._fetch_detailed_information()

            return self.models
        except Exception as e:
            raise OllamaLibraryError(f"Scraping failed: {e}")

    def _scrape_page(self, url: str) -> None:
        """Recursively scrape a page and follow pagination links.

        Implements depth-first traversal of the Ollama library, extracting models
        from the current page before following pagination links to discover more.
        Tracks visited URLs to prevent cycles and redundant requests.

        Args:
            url: URL of the page to scrape
        """
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)
        html_content = self.http_client.get(url)
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract models from the current page
        model_cards = soup.select(self.config.card_selector)
        for card in self._safe_iterate(model_cards):
            model = self._extract_model_from_card(card)
            if model:
                self.models.append(model)

        # Follow pagination links to discover more models
        pagination_links = soup.select("a.pagination-link")
        for link in self._safe_iterate(pagination_links):
            href = link.get("href")
            if href and href not in self.visited_urls:
                next_url = self.config.model_url(href)
                self._scrape_page(next_url)  # Recursive traversal

    def _extract_model_from_card(self, card: Tag) -> Optional[OllamaModel]:
        """Extract model data from a card element.

        Parses a model card HTML element to extract model information,
        handling missing or malformed data gracefully.

        Args:
            card: Beautiful Soup Tag representing a model card

        Returns:
            Optional[OllamaModel]: Constructed model object or None if extraction fails
        """
        try:
            # Find the anchor tag inside the card
            anchor = card.select_one("a")
            if not anchor:
                return None

            # Extract href and generate URL
            href = anchor.get("href", "")
            if not href:
                return None

            url = self.config.model_url(href)

            # Extract model name from href path
            model_name = href.split("/")[-1] if "/" in href else href

            # Extract description with fallback to card-level element
            desc_elem = anchor.select_one(
                self.config.description_selector
            ) or card.select_one(self.config.description_selector)
            description = (
                desc_elem.text.strip() if desc_elem else "No description available"
            )

            # Extract additional metadata components
            tags = self._extract_tags(card)
            metadata = ModelMetadata(
                size=self._extract_size(card), parameters=self._extract_parameters(card)
            )

            return OllamaModel(
                name=model_name,
                description=description,
                url=url,
                tags=tags,
                metadata=metadata,
            )
        except Exception as e:
            # Log extraction failure but continue with other models
            print(f"⚠️ Failed to extract model: {e}")
            return None

    def _extract_tags(self, card: Tag) -> List[str]:
        """Extract tag information from a model card.

        Args:
            card: Beautiful Soup Tag representing a model card

        Returns:
            List[str]: Collection of tag strings associated with the model
        """
        tag_elems = card.select(self.config.tag_selector)
        return [tag.text.strip() for tag in self._safe_iterate(tag_elems)]

    def _extract_size(self, card: Tag) -> Optional[str]:
        """Extract model size information from a card.

        Args:
            card: Beautiful Soup Tag representing a model card

        Returns:
            Optional[str]: Size string if available, None otherwise
        """
        size_elem = card.select_one(self.config.size_selector)
        return size_elem.text.strip() if size_elem else None

    def _extract_parameters(self, card: Tag) -> Optional[str]:
        """Extract model parameter count information from a card.

        Args:
            card: Beautiful Soup Tag representing a model card

        Returns:
            Optional[str]: Parameter count string if available, None otherwise
        """
        params_elem = card.select_one(self.config.parameter_selector)
        return params_elem.text.strip() if params_elem else None

    def _fetch_detailed_information(self) -> None:
        """Fetch detailed information for all models concurrently.

        Uses appropriate concurrency mechanism based on context - avoiding
        nested event loops when called from an already running loop.
        """
        # Check if we're already in an event loop
        try:
            asyncio.get_running_loop()
            # We're in an event loop, use a ThreadPoolExecutor instead
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.concurrent_requests
            ) as executor:
                # Execute detail extraction concurrently using threads
                futures = [
                    executor.submit(self.detail_extractor.extract_model_details, model)
                    for model in self.models
                ]
                concurrent.futures.wait(futures)  # Wait for all extractions to complete
        except RuntimeError:
            # Not in an event loop, proceed with normal asyncio approach
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # Create a new event loop if there isn't one in this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Create tasks for all models and execute them concurrently
            tasks = [self._fetch_model_details(model) for model in self.models]
            loop.run_until_complete(asyncio.gather(*tasks))

    async def _fetch_model_details(self, model: OllamaModel) -> None:
        """Asynchronously fetch detailed information for a single model.

        Uses a semaphore to limit concurrent requests and delegates to
        the detail extractor for the actual information retrieval.

        Args:
            model: The model to fetch details for
        """
        async with self.semaphore:
            # Execute the synchronous detail extractor in a thread pool
            # to prevent blocking the event loop
            await asyncio.get_event_loop().run_in_executor(
                None, self.detail_extractor.extract_model_details, model
            )

    @staticmethod
    def _safe_iterate(
        items: Union[ResultSet, List[Tag], Iterator[Tag]],
    ) -> Iterator[Tag]:
        """Safely iterate through Beautiful Soup elements, handling None values.

        Provides a defensive iteration mechanism that filters out None values
        and handles empty collections gracefully.

        Args:
            items: Iterable of Beautiful Soup Tags

        Yields:
            Tag: Each valid Tag element
        """
        if not items:
            return
        for item in items:
            if item:
                yield item


class ModelIndexer:
    """Manages local indexing of Ollama models for efficient searching and retrieval.

    Provides disk persistence with dual-index architecture: a basic index for backwards
    compatibility and a detailed index for complete model information. Handles all
    serialization, deserialization, and validation with robust error handling.

    Attributes:
        index_path: Path to the basic JSON index file.
        detailed_index_path: Path to the detailed model information file.
    """

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        detailed_index_path: str = DEFAULT_DETAILED_INDEX_PATH,
    ) -> None:
        """Initialize indexer with paths to index files.

        Args:
            index_path: Path to basic JSON index file. Defaults to DEFAULT_INDEX_PATH.
            detailed_index_path: Path to detailed model information file.
                Defaults to DEFAULT_DETAILED_INDEX_PATH.
        """
        self.index_path: str = index_path
        self.detailed_index_path: str = detailed_index_path

    def save(self, models: List[OllamaModel]) -> None:
        """Save models to local index files.

        Persists model data to both a basic index (for backward compatibility)
        and a detailed index with complete model information. Ensures parent
        directories exist before writing.

        Args:
            models: List of OllamaModel objects to save.

        Raises:
            IndexingError: If directory creation or file writing fails.
        """
        try:
            # Ensure parent directories exist for both index paths
            for path in [self.index_path, self.detailed_index_path]:
                directory = os.path.dirname(os.path.abspath(path))
                os.makedirs(directory, exist_ok=True)

            # Convert models to serializable dictionaries
            model_dicts: List[ModelData] = [model.to_dict() for model in models]

            # Save the basic models index (compatibility with older versions)
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(model_dicts, f, ensure_ascii=False, indent=2)

            # Save the detailed models with all metadata
            with open(self.detailed_index_path, "w", encoding="utf-8") as f:
                json.dump(model_dicts, f, ensure_ascii=False, indent=2)

            print(f"✅ Model indexes updated successfully with {len(models)} models.")
        except (IOError, json.JSONDecodeError, OSError) as e:
            raise IndexingError(f"Failed to save model index: {e}")

    def load(self) -> List[OllamaModel]:
        """Load models from local index files.

        Attempts to load from the detailed index first, falling back to the basic
        index if the detailed one is unavailable or corrupted.

        Returns:
            List[OllamaModel]: List of parsed OllamaModel objects.

        Raises:
            IndexingError: If loading fails from both index files or if no index exists.
        """
        # First attempt: Try loading from the detailed index
        if os.path.exists(self.detailed_index_path):
            try:
                with open(self.detailed_index_path, "r", encoding="utf-8") as f:
                    return self._parse_model_dicts(json.load(f))
            except (IOError, json.JSONDecodeError) as e:
                print(f"⚠️ Warning: Could not load detailed index: {e}")
                # Continue to fallback option

        # Second attempt: Fall back to the basic index
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    return self._parse_model_dicts(json.load(f))
            except (IOError, json.JSONDecodeError) as e:
                raise IndexingError(f"Failed to load model index: {e}")

        raise IndexingError("No model index found. Please run an update first.")

    def _parse_model_dicts(self, model_dicts: List[ModelData]) -> List[OllamaModel]:
        """Parse model dictionaries into structured OllamaModel objects.

        Converts raw dictionary data loaded from JSON into properly typed
        OllamaModel instances with full metadata.

        Args:
            model_dicts: List of dictionaries containing model data.

        Returns:
            List[OllamaModel]: List of properly constructed OllamaModel objects.
        """
        models: List[OllamaModel] = []

        for model_dict in model_dicts:
            try:
                # Extract basic model information
                name = str(model_dict.get("name", ""))
                description = str(model_dict.get("description", ""))
                url = str(model_dict.get("url", ""))

                # Extract tags, ensuring they're strings
                tags_data = model_dict.get("tags", [])
                tags: List[str] = [
                    str(tag)
                    for tag in (tags_data if isinstance(tags_data, list) else [])
                ]

                # Extract metadata
                metadata_dict = model_dict.get("metadata", {})
                if not isinstance(metadata_dict, dict):
                    metadata_dict = {}

                metadata = ModelMetadata(
                    size=metadata_dict.get("size"),
                    parameters=metadata_dict.get("parameters"),
                    license=metadata_dict.get("license"),
                    author=metadata_dict.get("author"),
                    last_updated=metadata_dict.get("last_updated"),
                    download_count=metadata_dict.get("download_count"),
                    capabilities=cast(List[str], metadata_dict.get("capabilities", [])),
                    languages=cast(List[str], metadata_dict.get("languages", [])),
                    context_length=metadata_dict.get("context_length"),
                    quantization=metadata_dict.get("quantization"),
                )

                # Create the model object
                model = OllamaModel(
                    name=name,
                    description=description,
                    url=url,
                    tags=tags,
                    metadata=metadata,
                )

                models.append(model)
            except Exception as e:
                print(f"⚠️ Warning: Could not parse model: {e}")
                continue

        return models


class ModelSearcher:
    """Advanced search capabilities for Ollama models.

    Provides semantic search functionality with scoring based on relevance and
    various filter mechanisms for precise model discovery.
    """

    @staticmethod
    def search(
        models: List[OllamaModel],
        query: str,
        filter_funcs: Optional[List[ModelFilter]] = None,
    ) -> List[OllamaModel]:
        """Search models by name, description, or tags with optional filters.

        Performs a multi-field search across model attributes with intelligent
        scoring that prioritizes name matches over tag matches, and tag matches
        over description matches.

        Args:
            models: List of models to search.
            query: Search query string.
            filter_funcs: Optional list of filter functions to apply.

        Returns:
            List[OllamaModel]: List of matching models ordered by relevance.
        """
        if not query.strip() and not filter_funcs:
            return models

        # Apply filters first if provided
        filtered_models = models
        if filter_funcs:
            filtered_models = [
                model
                for model in filtered_models
                if all(model.satisfies_filter(f) for f in filter_funcs)
            ]

        # If there's no query, return the filtered models
        if not query.strip():
            return filtered_models

        # Process the query terms
        query_terms = re.split(r"\s+", query.lower())
        results: List[Tuple[int, OllamaModel]] = []

        for model in filtered_models:
            score = ModelSearcher._calculate_match_score(model, query_terms)
            if score > 0:
                results.append((score, model))

        # Sort by score, highest first
        results.sort(reverse=True, key=lambda x: x[0])
        return [model for _, model in results]

    @staticmethod
    def _calculate_match_score(model: OllamaModel, query_terms: List[str]) -> int:
        """Calculate match score for a model against query terms.

        Uses a weighted scoring system that prioritizes name matches, then tags,
        then descriptions. Also gives bonus points for prefix matches.

        Args:
            model: Model to score.
            query_terms: List of query terms to match against.

        Returns:
            int: Match score (higher is better match).
        """
        score = 0
        name_lower = model.name.lower()
        desc_lower = model.description.lower()
        tags_lower = [tag.lower() for tag in model.tags]

        # Also search in metadata
        metadata_text = ""
        if model.metadata.parameters:
            metadata_text += f" {model.metadata.parameters}"
        if model.metadata.size:
            metadata_text += f" {model.metadata.size}"
        if model.metadata.context_length:
            metadata_text += f" {model.metadata.context_length}"
        if model.metadata.capabilities:
            metadata_text += f" {' '.join(model.metadata.capabilities)}"
        if model.metadata.languages:
            metadata_text += f" {' '.join(model.metadata.languages)}"
        metadata_lower = metadata_text.lower()

        for term in query_terms:
            # Exact matches in name are highest priority
            if term in name_lower:
                score += 5
                if name_lower.startswith(term):
                    score += 3  # Bonus for prefix match
                elif name_lower == term:
                    score += 5  # Bonus for exact match

            # Matches in tags are second priority
            if any(term in tag for tag in tags_lower):
                score += 4
                if any(tag == term for tag in tags_lower):
                    score += 2  # Bonus for exact tag match

            # Matches in metadata
            if term in metadata_lower:
                score += 3

            # Matches in description
            if term in desc_lower:
                score += 2  # Lower priority than metadata

        return score

    @staticmethod
    def create_parameter_filter(
        min_params: Optional[float] = None, max_params: Optional[float] = None
    ) -> ModelFilter:
        """Create a filter function for model parameter count.

        Args:
            min_params: Minimum parameter count in billions.
            max_params: Maximum parameter count in billions.

        Returns:
            ModelFilter: Filter function for parameter count.
        """

        def filter_func(model: OllamaModel) -> bool:
            if not model.metadata.parameters:
                return False

            # Extract numeric value from parameters string
            param_str = model.metadata.parameters.lower()
            matches = re.search(r"(\d+(\.\d+)?)", param_str)
            if not matches:
                return False

            try:
                param_value = float(matches.group(1))
                # Adjust for units (M vs B)
                if "m" in param_str:
                    param_value /= 1000  # Convert M to B

                # Apply filters
                if min_params is not None and param_value < min_params:
                    return False
                if max_params is not None and param_value > max_params:
                    return False
                return True
            except ValueError:
                return False  # Return False on parsing error

        return filter_func

    @staticmethod
    def create_tag_filter(required_tags: List[str]) -> ModelFilter:
        """Create a filter function for model tags.

        Args:
            required_tags: List of tags that must be present.

        Returns:
            ModelFilter: Filter function for tags.
        """
        required_lower = [tag.lower() for tag in required_tags]

        def filter_func(model: OllamaModel) -> bool:
            model_tags_lower = [tag.lower() for tag in model.tags]
            return all(tag in model_tags_lower for tag in required_lower)

        return filter_func

    @staticmethod
    def create_context_length_filter(
        min_length: Optional[int] = None,
    ) -> ModelFilter:
        """Create a filter function for model context length.

        Args:
            min_length: Minimum context length in tokens.

        Returns:
            ModelFilter: Filter function for context length.
        """

        def filter_func(model: OllamaModel) -> bool:
            if not model.metadata.context_length:
                return False

            # Extract numeric value from context length string
            context_str = model.metadata.context_length.lower()
            matches = re.search(r"(\d+)", context_str)
            if not matches:
                return False

            try:
                context_value = int(matches.group(1))
                # Adjust for units (K)
                if "k" in context_str:
                    context_value *= 1000

                # Apply filter
                return min_length is None or context_value >= min_length
            except ValueError:
                return False

        return filter_func


def display_models(models: List[OllamaModel]) -> None:
    """Display models in a human-readable format with available metadata.

    Presents the models with all their metadata in a visually appealing format.

    Args:
        models: List of models to display.
    """
    if not models:
        print("No models found.")
        return

    print(f"\n📦 Found {len(models)} model(s):")
    for model in models:
        print(f"🔎 Name: {model.name}")
        print(f"📝 Description: {model.description}")
        print(f"🌐 URL: {model.url}")

        if model.tags:
            print(f"🏷️ Tags: {', '.join(model.tags)}")

        # Display all available metadata
        metadata = model.metadata
        if metadata.size:
            print(f"💾 Size: {metadata.size}")

        if metadata.parameters:
            print(f"🧮 Parameters: {metadata.parameters}")

        print("-" * 50)


def main() -> None:
    """Main function orchestrating the Ollama Library scraper workflow."""
    print("🧠 Eidosian Ollama Library Explorer 🤖")

    # Initialize components with default configuration
    config = ScraperConfig()
    scraper = OllamaLibraryScraper(config)
    indexer = ModelIndexer(config.index_path)

    while True:
        print("\nOptions:")
        print("1. Update local model index (scrape library)")
        print("2. Search for models")
        print("3. Exit")
        choice = input("Select an option (1/2/3): ").strip()

        if choice == "1":
            try:
                print("🔄 Scraping Ollama Library for models...")
                models = scraper.scrape()
                indexer.save(models)
            except (OllamaLibraryError, IndexingError) as e:
                print(f"❌ Error: {e}")

        elif choice == "2":
            try:
                models = indexer.load()
                query = input("🔍 Enter search query: ").strip()
                results = ModelSearcher.search(models, query)
                display_models(results)
            except IndexingError as e:
                print(f"❌ Error: {e}")

        elif choice == "3":
            print("👋 Goodbye!")
            break

        else:
            print("❓ Invalid option. Please try again.")


if __name__ == "__main__":
    main()
