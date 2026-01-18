"""
Eidosian Model Manager
======================

Central coordination system for multi-source model management with unified interfaces
for searching, installing, and managing models from Ollama and Hugging Face.

Key Features:
- Universal search across all configured model sources
- Unified model installation regardless of source
- Efficient local model management with source-specific handlers
- Extensible architecture for adding new model sources
- Complete type safety with Protocol interfaces
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import requests
from requests.exceptions import RequestException

from ollama_forge.src.ollama_forge.models import (
    ModelInfo,
    ModelInstaller,
    ModelProvider,
    ModelSize,
    ModelSource,
)
from ollama_forge.src.ollama_forge.models.browse_remote_huggingface_models import (
    fetch_models as fetch_huggingface_models,
)
from ollama_forge.src.ollama_forge.models.browse_remote_ollama_models import (
    ModelIndexer,
    OllamaLibraryError,
    OllamaLibraryScraper,
    ScraperConfig,
)
from ollama_forge.src.ollama_forge.models.list_local_ollama_models import (
    OllamaLocalModelProvider,
)

# Type aliases
T = TypeVar("T")
SearchResult = Tuple[int, ModelInfo]  # Score, Model
SearchResults = List[SearchResult]
ConfigDict = Dict[str, Any]


class OllamaRemoteModelProvider(ModelProvider):
    """Provider for remote Ollama models from the Ollama Library."""

    def __init__(self, index_path: str):
        """Initialize the Ollama remote model provider with index path.

        Args:
            index_path: Path to the model index file
        """
        self.index_path = index_path
        self.indexer = ModelIndexer(index_path)
        self.models: List[ModelInfo] = []

        # Try to load existing index
        try:
            self._load_models()
        except Exception:
            # Initialize with empty model list if index doesn't exist
            self.models = []

    def _load_models(self) -> None:
        """Load models from the index file."""
        try:
            ollama_models = self.indexer.load()
            self.models = [
                self._convert_to_model_info(model) for model in ollama_models
            ]
        except Exception as e:
            print(f"⚠️ Warning: Could not load Ollama remote model index: {e}")
            self.models = []

    def _convert_to_model_info(self, ollama_model: Any) -> ModelInfo:
        """Convert an OllamaModel to ModelInfo.

        Args:
            ollama_model: OllamaModel object from the indexer

        Returns:
            ModelInfo: Converted model information
        """
        # Extract metadata from the OllamaModel
        metadata = ollama_model.metadata

        # Create ModelInfo with extracted data
        model_info = ModelInfo(
            name=ollama_model.name,
            source=ModelSource.OLLAMA_REMOTE,
            description=ollama_model.description,
            tags=ollama_model.tags,
            url=ollama_model.url,
            installed=False,  # Remote models are not installed by default
        )

        # Add size if available
        if metadata.size:
            model_info.size = ModelSize.parse(metadata.size)

        # Add parameters if available
        if metadata.parameters:
            model_info.parameters = metadata.parameters

        # Add context length if available
        if metadata.context_length:
            try:
                # Try to parse context length to int if possible
                model_info.context_length = int(
                    metadata.context_length.replace("k", "000")
                )
            except (ValueError, AttributeError):
                # Store as is if parsing fails
                model_info.context_length = None

        # Add quantization if available
        if metadata.quantization:
            model_info.quantization = metadata.quantization

        # Store all metadata for future reference
        model_info.metadata = {
            "license": metadata.license,
            "author": metadata.author,
            "last_updated": metadata.last_updated,
            "download_count": metadata.download_count,
            "capabilities": metadata.capabilities,
            "languages": metadata.languages,
        }

        return model_info

    def list_models(
        self, query: Optional[str] = None, installed_only: bool = False, **kwargs
    ) -> List[ModelInfo]:
        """List available models from Ollama remote library.

        Args:
            query: Optional search query
            installed_only: If True, only return installed models
            **kwargs: Additional source-specific parameters

        Returns:
            List[ModelInfo]: List of available models
        """
        # If we're looking for installed models only, return empty list since remote models
        # aren't installed by default
        if installed_only:
            return []

        # Make sure we have models loaded
        if not self.models:
            self._load_models()

        # If still no models, return empty list
        if not self.models:
            return []

        # Filter by query if provided
        if query:
            query_lower = query.lower()
            filtered_models = []

            for model in self.models:
                # Check if query appears in name, description, or tags
                name_match = query_lower in model.name.lower()
                desc_match = (
                    model.description and query_lower in model.description.lower()
                )
                tag_match = any(query_lower in tag.lower() for tag in model.tags)

                if name_match or desc_match or tag_match:
                    filtered_models.append(model)

            return filtered_models

        # Return all models if no query
        return self.models

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model.

        Args:
            name: Model name to retrieve

        Returns:
            Optional[ModelInfo]: Model information or None if not found
        """
        # Make sure we have models loaded
        if not self.models:
            self._load_models()

        # Search for the model by name
        for model in self.models:
            if model.name == name:
                return model

        return None

    async def update_index(self) -> None:
        """Update the model index by scraping the Ollama Library.

        This method scrapes the Ollama Library website and updates the local index.
        """
        try:
            print("🔄 Updating Ollama remote model index...")

            # Configure and execute the scraper
            config = ScraperConfig(
                index_path=self.index_path,
                detailed_index_path=self.index_path.replace(".json", "_detailed.json"),
                fetch_detailed_info=True,  # Get complete model information
            )

            scraper = OllamaLibraryScraper(config)
            ollama_models = scraper.scrape()

            # Save models to index
            self.indexer.save(ollama_models)

            # Reload the models
            self._load_models()

            print(f"✅ Ollama remote index updated with {len(ollama_models)} models")
        except OllamaLibraryError as e:
            print(f"❌ Error updating Ollama remote index: {e}")
            raise


class HuggingFaceModelProvider(ModelProvider):
    """Provider for Hugging Face models using the Hugging Face API."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the Hugging Face model provider.

        Args:
            cache_dir: Optional directory to cache model information
        """
        self.cache_dir = cache_dir
        self.cache: Dict[str, ModelInfo] = {}

    def list_models(
        self, query: Optional[str] = None, installed_only: bool = False, **kwargs
    ) -> List[ModelInfo]:
        """List available models from Hugging Face.

        Args:
            query: Optional search query
            installed_only: If True, only return installed models
            **kwargs: Additional source-specific parameters

        Returns:
            List[ModelInfo]: List of available models
        """
        # Hugging Face models aren't locally installed, so return empty list for installed_only
        if installed_only:
            return []

        try:
            # Set reasonable limits to avoid overloading the API
            limit = min(kwargs.get("limit", 50), 100)

            # Determine the pipeline filter for text generation models
            filters = {"pipeline_tag": "text-generation"}

            # Merge with any additional filters
            if "filters" in kwargs and isinstance(kwargs["filters"], dict):
                filters.update(kwargs["filters"])

            # Fetch models from Hugging Face API
            hf_models = fetch_huggingface_models(
                query=query, limit=limit, filters=filters
            )

            # Convert to ModelInfo objects
            models: List[ModelInfo] = []
            for hf_model in hf_models:
                model_info = self._convert_to_model_info(hf_model)
                models.append(model_info)

                # Cache for future get_model requests
                self.cache[model_info.name] = model_info

            return models

        except Exception as e:
            print(f"⚠️ Error fetching models from Hugging Face: {e}")
            return []

    def _convert_to_model_info(self, hf_model: Dict[str, Any]) -> ModelInfo:
        """Convert a Hugging Face model dict to ModelInfo.

        Args:
            hf_model: Dictionary with Hugging Face model data

        Returns:
            ModelInfo: Converted model information
        """
        model_id = str(hf_model.get("id", ""))

        # Create model tags from available data
        tags = []
        if "pipeline_tag" in hf_model:
            tags.append(str(hf_model["pipeline_tag"]))

        if "tags" in hf_model and isinstance(hf_model["tags"], list):
            tags.extend([str(tag) for tag in hf_model["tags"]])

        # Get model URL
        url = f"https://huggingface.co/{model_id}"

        # Create the ModelInfo
        model_info = ModelInfo(
            name=model_id,
            source=ModelSource.HUGGINGFACE,
            description=str(hf_model.get("description", "No description available")),
            tags=tags,
            url=url,
            installed=False,
        )

        # Store all metadata for future reference
        model_info.metadata = {
            "author": hf_model.get("author"),
            "downloads": hf_model.get("downloads"),
            "likes": hf_model.get("likes"),
            "created_at": hf_model.get("created_at"),
            "last_modified": hf_model.get("last_modified"),
            "private": hf_model.get("private", False),
        }

        return model_info

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model.

        Args:
            name: Model name to retrieve (Hugging Face model ID)

        Returns:
            Optional[ModelInfo]: Model information or None if not found
        """
        # Check if model is in cache first
        if name in self.cache:
            return self.cache[name]

        try:
            # Use the API to get model details (singular lookup)
            url = f"https://huggingface.co/api/models/{name}"
            response = requests.get(url)

            if response.status_code != 200:
                return None

            hf_model = response.json()
            model_info = self._convert_to_model_info(hf_model)

            # Cache for future requests
            self.cache[name] = model_info

            return model_info

        except RequestException:
            return None


class ModelManager:
    """Core manager for unified model operations across multiple sources.

    Provides a single interface for discovering, installing, and managing
    models from multiple sources including Ollama and Hugging Face.

    Attributes:
        providers: Dictionary mapping model sources to provider implementations
        installers: Dictionary mapping model sources to installer implementations
        installed_models: Cache of currently installed models
        config: Configuration settings for the manager
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the model manager with optional configuration.

        Args:
            config_path: Path to configuration file. If None, default config is used.
        """
        self.providers: Dict[ModelSource, ModelProvider] = {}
        self.installers: Dict[ModelSource, ModelInstaller] = {}

        # Cache of installed models
        self.installed_models: Dict[str, ModelInfo] = {}

        # Initialize default configuration
        self.config: ConfigDict = self._load_config(config_path)

        # Register default providers and installers
        self._register_default_providers()

    def _load_config(self, config_path: Optional[str]) -> ConfigDict:
        """Load configuration from file or use defaults.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary with all settings
        """
        default_config: ConfigDict = {
            "ollama_api_url": "http://localhost:11434/api",
            "cache_dir": os.path.expanduser("~/.ollama_forge/cache"),
            "index_path": os.path.expanduser("~/.ollama_forge/model_index.json"),
        }

        if not config_path or not os.path.exists(config_path):
            return default_config

        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
                # Update defaults with user configuration
                default_config.update(user_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Warning: Could not load config from {config_path}: {e}")

        return default_config

    def _register_default_providers(self) -> None:
        """Register the default model providers and installers.

        Sets up Ollama local, Ollama remote, and Hugging Face providers
        with appropriate paths and configurations.
        """
        # Ensure cache directory exists
        os.makedirs(self.config["cache_dir"], exist_ok=True)

        # Register Ollama local provider
        ollama_local = OllamaLocalModelProvider(
            api_base_url=self.config["ollama_api_url"]
        )
        self.register_provider(ModelSource.OLLAMA_LOCAL, ollama_local)
        self.register_installer(ModelSource.OLLAMA_LOCAL, ollama_local)

        # Register Ollama remote provider
        ollama_remote = OllamaRemoteModelProvider(
            index_path=os.path.join(
                self.config["cache_dir"], "ollama_remote_index.json"
            )
        )
        self.register_provider(ModelSource.OLLAMA_REMOTE, ollama_remote)

        # Register Hugging Face provider
        huggingface = HuggingFaceModelProvider(cache_dir=self.config["cache_dir"])
        self.register_provider(ModelSource.HUGGINGFACE, huggingface)

    def register_provider(self, source: ModelSource, provider: ModelProvider) -> None:
        """Register a model provider for a specific source.

        Args:
            source: The model source identifier
            provider: The provider implementation
        """
        self.providers[source] = provider

    def register_installer(
        self, source: ModelSource, installer: ModelInstaller
    ) -> None:
        """Register a model installer for a specific source.

        Args:
            source: The model source identifier
            installer: The installer implementation
        """
        self.installers[source] = installer

    def search_models(
        self,
        query: str = "",
        sources: Optional[List[ModelSource]] = None,
        **kwargs: Any,
    ) -> List[ModelInfo]:
        """Search for models across multiple sources.

        Args:
            query: Search query string to filter models
            sources: List of sources to search, or None for all sources
            **kwargs: Additional source-specific parameters

        Returns:
            List of matching models across all requested sources
        """
        if sources is None:
            sources = list(self.providers.keys())

        # First, refresh the cache of installed models
        self.refresh_installed_models()

        # Search across all requested sources
        all_results: List[ModelInfo] = []
        for source in sources:
            if source not in self.providers:
                print(f"⚠️ Warning: No provider registered for source {source}")
                continue

            provider = self.providers[source]
            try:
                models = provider.list_models(query=query, **kwargs)

                # Mark models as installed if they're in our installed cache
                for model in models:
                    if model.name in self.installed_models:
                        model.installed = True

                all_results.extend(models)
            except Exception as e:
                print(f"⚠️ Error searching in {source}: {e}")

        # Sort results by relevance to query
        if query:
            scored_results = self._score_search_results(query, all_results)
            all_results = [model for _, model in scored_results]

        return all_results

    def _score_search_results(
        self, query: str, models: List[ModelInfo]
    ) -> SearchResults:
        """Score and sort models based on relevance to query.

        Uses a multi-factor scoring algorithm considering name, tags,
        description, and installation status.

        Args:
            query: Search query
            models: List of models to score

        Returns:
            List of (score, model) tuples sorted by score (highest first)
        """
        query_terms = query.lower().split()
        scored_results: SearchResults = []

        for model in models:
            score = 0
            name_lower = model.name.lower()
            desc_lower = model.description.lower()
            tags_lower = [tag.lower() for tag in model.tags]

            # Score each query term
            for term in query_terms:
                # Exact name match is highest priority
                if term in name_lower:
                    score += 10
                    if name_lower.startswith(term):
                        score += 5  # Bonus for prefix match

                # Tag matches are second priority
                if any(term in tag for tag in tags_lower):
                    score += 8

                # Description matches are lower priority
                if term in desc_lower:
                    score += 5

            # Boost installed models
            if model.installed:
                score += 3

            # Only include models with a positive score
            if score > 0:
                scored_results.append((score, model))

        # Sort by score, highest first
        scored_results.sort(reverse=True, key=lambda x: x[0])
        return scored_results

    def get_model_details(self, name: str, source: ModelSource) -> Optional[ModelInfo]:
        """Get detailed information about a specific model.

        Args:
            name: Model name to fetch
            source: Source to search in

        Returns:
            Model details or None if not found
        """
        if source not in self.providers:
            print(f"⚠️ Warning: No provider registered for source {source}")
            return None

        provider = self.providers[source]
        model = provider.get_model(name)

        # Check if the model is installed
        if model and name in self.installed_models:
            model.installed = True

        return model

    def install_model(self, model: ModelInfo) -> bool:
        """Install a model from its source.

        Args:
            model: Model information object to install

        Returns:
            True if installation was successful, False otherwise
        """
        if model.source not in self.installers:
            print(f"⚠️ Error: No installer available for source {model.source}")
            return False

        installer = self.installers[model.source]
        success = installer.install_model(model)

        if success:
            # Update our cache of installed models
            self.refresh_installed_models()
            print(f"✅ Successfully installed {model.name}")
        else:
            print(f"❌ Failed to install {model.name}")

        return success

    def uninstall_model(self, model_name: str) -> bool:
        """Uninstall a model.

        Args:
            model_name: Name of the model to uninstall

        Returns:
            True if uninstallation was successful, False otherwise
        """
        # Find the model in our installed cache
        if model_name not in self.installed_models:
            print(f"⚠️ Model {model_name} is not installed")
            return False

        model = self.installed_models[model_name]

        if model.source not in self.installers:
            print(f"⚠️ Error: No installer available for source {model.source}")
            return False

        installer = self.installers[model.source]
        success = installer.uninstall_model(model_name)

        if success:
            # Update our cache of installed models
            self.refresh_installed_models()
            print(f"✅ Successfully uninstalled {model_name}")
        else:
            print(f"❌ Failed to uninstall {model_name}")

        return success

    def list_installed_models(self) -> List[ModelInfo]:
        """List all installed models across all sources.

        Returns:
            List of installed model information objects
        """
        self.refresh_installed_models()
        return list(self.installed_models.values())

    def refresh_installed_models(self) -> None:
        """Refresh the cache of installed models from all sources.

        Updates the internal cache of installed models by querying
        each provider that has an installer.
        """
        self.installed_models.clear()

        # Get installed models from each provider with an installer
        for source, provider in self.providers.items():
            if source not in self.installers:
                continue

            try:
                # We specifically want local/installed models here
                models = provider.list_models(installed_only=True)
                for model in models:
                    model.installed = True
                    self.installed_models[model.name] = model
            except Exception as e:
                print(f"⚠️ Error refreshing installed models from {source}: {e}")

    async def update_model_indices(self) -> None:
        """Update all remote model indices asynchronously.

        Refreshes model catalogs from remote sources in parallel
        for providers that support index updates.
        """
        tasks = []

        # Collect update tasks from providers that support it
        for source, provider in self.providers.items():
            if hasattr(provider, "update_index"):
                task = asyncio.create_task(provider.update_index())
                tasks.append(task)

        # Wait for all updates to complete
        if tasks:
            await asyncio.gather(*tasks)
            print("✅ All model indices updated successfully")

    def save_state(self) -> None:
        """Save current state to disk.

        Persists the current state of installed models to the configured
        state file location.
        """
        try:
            state = {
                "installed_models": [
                    asdict(model) for model in self.installed_models.values()
                ]
            }

            with open(self.config["index_path"], "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving state: {e}")

    def load_state(self) -> None:
        """Load state from disk.

        Restores the cached state of installed models from the
        previously saved state file.
        """
        if not os.path.exists(self.config["index_path"]):
            return

        try:
            with open(self.config["index_path"], "r") as f:
                state = json.load(f)

            # Process installed models
            if "installed_models" in state:
                self.installed_models = {}
                for model_dict in state["installed_models"]:
                    model = ModelInfo(**model_dict)
                    model.installed = True
                    self.installed_models[model.name] = model
        except Exception as e:
            print(f"⚠️ Error loading state: {e}")


def create_cli() -> None:
    """Create a CLI interface for the model manager.

    Sets up an argparse-based command-line interface with commands for
    searching, listing, installing, and uninstalling models.
    """
    parser = argparse.ArgumentParser(description="Eidosian Model Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for models")
    search_parser.add_argument("query", nargs="?", default="", help="Search query")
    search_parser.add_argument(
        "--source",
        choices=["all", "ollama-local", "ollama-remote", "huggingface"],
        default="all",
        help="Source to search in",
    )

    # List command
    subparsers.add_parser("list", help="List installed models")

    # Install command
    install_parser = subparsers.add_parser("install", help="Install a model")
    install_parser.add_argument("name", help="Model name to install")
    install_parser.add_argument(
        "--source",
        choices=["ollama-local", "ollama-remote", "huggingface"],
        required=True,
        help="Source to install from",
    )

    # Uninstall command
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall a model")
    uninstall_parser.add_argument("name", help="Model name to uninstall")

    # Update index command
    subparsers.add_parser("update-index", help="Update model indices")

    # Parse arguments
    args = parser.parse_args()

    # Create model manager
    manager = ModelManager()

    # Execute command
    if args.command == "search":
        _handle_search_command(args, manager)
    elif args.command == "list":
        _handle_list_command(manager)
    elif args.command == "install":
        _handle_install_command(args, manager)
    elif args.command == "uninstall":
        _handle_uninstall_command(args, manager)
    elif args.command == "update-index":
        _handle_update_index_command(manager)
    else:
        parser.print_help()


def _handle_search_command(args: argparse.Namespace, manager: ModelManager) -> None:
    """Handle the search command.

    Args:
        args: Parsed command line arguments
        manager: Model manager instance
    """
    # Map source string to ModelSource enum
    sources: List[ModelSource] = []
    if args.source == "all":
        sources = list(ModelSource)
    elif args.source == "ollama-local":
        sources = [ModelSource.OLLAMA_LOCAL]
    elif args.source == "ollama-remote":
        sources = [ModelSource.OLLAMA_REMOTE]
    elif args.source == "huggingface":
        sources = [ModelSource.HUGGINGFACE]

    models = manager.search_models(args.query, sources)
    if not models:
        print("No models found.")
        return

    print(f"Found {len(models)} models:")
    for model in models:
        installed_mark = "📦 " if model.installed else "   "
        print(f"{installed_mark}{model.name} ({model.source.name})")
        print(f"   {model.description[:80]}...")
        if model.size:
            print(f"   Size: {model.size}")
        if model.parameters:
            print(f"   Parameters: {model.parameters}")
        print()


def _handle_list_command(manager: ModelManager) -> None:
    """Handle the list command.

    Args:
        manager: Model manager instance
    """
    models = manager.list_installed_models()
    if not models:
        print("No models installed.")
        return

    print(f"Installed models ({len(models)}):")
    for model in models:
        print(f"📦 {model.name} ({model.source.name})")
        print(f"   {model.description[:80]}...")
        if model.size:
            print(f"   Size: {model.size}")
        print()


def _handle_install_command(args: argparse.Namespace, manager: ModelManager) -> None:
    """Handle the install command.

    Args:
        args: Parsed command line arguments
        manager: Model manager instance
    """
    # Map source string to ModelSource enum
    source: Optional[ModelSource] = None
    if args.source == "ollama-local":
        source = ModelSource.OLLAMA_LOCAL
    elif args.source == "ollama-remote":
        source = ModelSource.OLLAMA_REMOTE
    elif args.source == "huggingface":
        source = ModelSource.HUGGINGFACE

    if not source:
        print(f"Invalid source: {args.source}")
        sys.exit(1)

    # Get model details
    model = manager.get_model_details(args.name, source)
    if not model:
        print(f"Model {args.name} not found in {args.source}")
        sys.exit(1)

    # Install model
    success = manager.install_model(model)
    if not success:
        sys.exit(1)


def _handle_uninstall_command(args: argparse.Namespace, manager: ModelManager) -> None:
    """Handle the uninstall command.

    Args:
        args: Parsed command line arguments
        manager: Model manager instance
    """
    success = manager.uninstall_model(args.name)
    if not success:
        sys.exit(1)


def _handle_update_index_command(manager: ModelManager) -> None:
    """Handle the update-index command.

    Args:
        manager: Model manager instance
    """
    try:
        # Run the update in an event loop
        asyncio.run(manager.update_model_indices())
    except Exception as e:
        print(f"Error updating indices: {e}")
        sys.exit(1)


if __name__ == "__main__":
    create_cli()
