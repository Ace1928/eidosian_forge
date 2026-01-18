"""
Eidosian Ollama Local Model Provider
====================================

Implementation of the ModelProvider and ModelInstaller protocols for local Ollama models.
Provides efficient access to locally installed models and handles installation/uninstallation
via the Ollama API and CLI.

Key Features:
- Complete ModelProvider implementation for local Ollama models
- Efficient ModelInstaller implementation for pull/delete operations
- Comprehensive model metadata extraction and parsing
- Clean error boundaries with explicit failure modes
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Union

import requests

from . import (
    DownloadProgress,
    ModelInfo,
    ModelInstaller,
    ModelProvider,
    ModelSize,
    ModelSource,
)

# Constants
DEFAULT_API_BASE_URL: str = "http://localhost:11434/api"


# Exceptions
class OllamaAPIError(Exception):
    """Raised when the Ollama API returns an error."""

    pass


class OllamaCommandError(Exception):
    """Raised when an Ollama CLI command fails."""

    pass


class ValidationError(Exception):
    """Raised for invalid input parameters."""

    pass


class ConsoleProgressReporter(DownloadProgress):
    """Simple console-based progress reporter."""

    def __init__(self, model_name: str):
        """Initialize with model name.

        Args:
            model_name: Name of the model being downloaded
        """
        self.model_name = model_name
        self.last_percentage: int = -1
        self.start_time = time.time()

    def update(self, bytes_downloaded: int, total_bytes: int) -> None:
        """Update and display download progress.

        Args:
            bytes_downloaded: Number of bytes downloaded
            total_bytes: Total bytes to download
        """
        if total_bytes > 0:
            percentage = int((bytes_downloaded / total_bytes) * 100)
            if percentage != self.last_percentage and percentage % 5 == 0:
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    speed = bytes_downloaded / elapsed / 1024 / 1024  # MB/s
                    print(
                        f"⏳ Downloading {self.model_name}: {percentage}% complete ({speed:.2f} MB/s)"
                    )
                    self.last_percentage = percentage
        elif bytes_downloaded > 0:
            # No total size known, just show downloaded bytes
            if time.time() - self.start_time > 2:  # Update every 2 seconds
                mb_downloaded = bytes_downloaded / 1024 / 1024
                print(
                    f"⏳ Downloading {self.model_name}: {mb_downloaded:.2f} MB downloaded"
                )
                self.start_time = time.time()


class OllamaLocalModelProvider(ModelProvider, ModelInstaller):
    """Provider for local Ollama models with installation capabilities."""

    def __init__(self, api_base_url: str = DEFAULT_API_BASE_URL):
        """Initialize the Ollama local model provider.

        Args:
            api_base_url: Base URL for the Ollama API
        """
        self.api_base_url = api_base_url
        self.tags_url = f"{api_base_url}/tags"
        self.pull_url = f"{api_base_url}/pull"
        self.delete_url = f"{api_base_url}/delete"

    def list_models(
        self, query: Optional[str] = None, installed_only: bool = False, **kwargs
    ) -> List[ModelInfo]:
        """List available local Ollama models.

        Args:
            query: Optional search query for filtering models
            installed_only: Only return installed models (always True for local)
            **kwargs: Additional parameters (ignored)

        Returns:
            List[ModelInfo]: List of available model information
        """
        try:
            # Fetch models from the Ollama API
            response = requests.get(self.tags_url)
            if response.status_code != 200:
                raise OllamaAPIError(
                    f"API Error: {response.status_code} - {response.text}"
                )

            data = response.json()
            models_data = data.get("models", [])

            # Convert to ModelInfo objects
            models: List[ModelInfo] = []
            for model_data in models_data:
                # Extract base model info
                name = model_data.get("name", "")

                # Skip if we have a query and it doesn't match
                if query and query.lower() not in name.lower():
                    continue

                # Extract model size from raw size bytes
                size_bytes = model_data.get("size", 0)
                size = self._format_size(size_bytes)

                # Build the model info
                model = ModelInfo(
                    name=name,
                    source=ModelSource.OLLAMA_LOCAL,
                    description=self._extract_description(model_data),
                    size=ModelSize.parse(size),
                    installed=True,  # Local models are always installed
                    # Store raw data for possible future use
                    metadata={
                        "digest": model_data.get("digest", ""),
                        "modified_at": model_data.get("modified_at", ""),
                        "size_bytes": size_bytes,
                    },
                )

                # Try to extract parameters from model name
                parameters = self._extract_parameters(name)
                if parameters:
                    model.parameters = parameters

                # Try to extract quantization from model name
                quantization = self._extract_quantization(name)
                if quantization:
                    model.quantization = quantization

                # Try to extract context length from model name or metadata
                context_length = self._extract_context_length(name)
                if context_length:
                    model.context_length = context_length

                # Extract tags from name (fallback)
                tags = self._extract_tags_from_name(name)
                if tags:
                    model.tags = tags

                models.append(model)

            return models

        except requests.RequestException as e:
            raise OllamaAPIError(f"Network Error: {e}")

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific local Ollama model.

        Args:
            name: Model name

        Returns:
            Optional[ModelInfo]: Model information or None if not found
        """
        # List all models and find the one with the matching name
        models = self.list_models()
        for model in models:
            if model.name == name:
                return model
        return None

    def install_model(
        self, model: ModelInfo, progress_callback: Optional[DownloadProgress] = None
    ) -> bool:
        """Install (pull) an Ollama model.

        Args:
            model: Model to install
            progress_callback: Optional callback for progress reporting

        Returns:
            bool: True if installation was successful
        """
        try:
            # Create default progress reporter if none provided
            if progress_callback is None:
                progress_callback = ConsoleProgressReporter(model.name)

            # For local models, we use the pull API
            # This is a streaming API, so we need to handle the response differently
            headers = {"Accept": "application/x-ndjson"}
            payload = {"name": model.name}

            # Stream the response and process each line
            with requests.post(
                self.pull_url, json=payload, headers=headers, stream=True
            ) as response:
                if response.status_code != 200:
                    raise OllamaAPIError(
                        f"Pull Error: {response.status_code} - {response.text}"
                    )

                total_bytes = 0
                downloaded_bytes = 0

                # Process each line from the streaming response
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "total" in data:
                                total_bytes = data["total"]
                            if "completed" in data:
                                downloaded_bytes = data["completed"]
                                progress_callback.update(downloaded_bytes, total_bytes)
                        except json.JSONDecodeError:
                            # Sometimes lines might not be valid JSON
                            pass

            return True

        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Error pulling model: {e}")
            return False

    def uninstall_model(self, model_name: str) -> bool:
        """Uninstall (delete) an Ollama model.

        Args:
            model_name: Name of the model to uninstall

        Returns:
            bool: True if uninstallation was successful
        """
        try:
            # For local models, we use the delete API
            payload = {"name": model_name}
            response = requests.delete(self.delete_url, json=payload)

            if response.status_code != 200:
                raise OllamaAPIError(
                    f"Delete Error: {response.status_code} - {response.text}"
                )

            return True

        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Error deleting model: {e}")
            return False

    def _extract_description(self, model_data: Dict[str, Any]) -> str:
        """Extract or generate a description for the model.

        Args:
            model_data: Raw model data from Ollama API

        Returns:
            str: Model description
        """
        # Try to extract from model data
        if "details" in model_data and isinstance(model_data["details"], dict):
            if "description" in model_data["details"]:
                return model_data["details"]["description"]

        # Generate a description based on the model name
        name = model_data.get("name", "")
        return f"Local Ollama model: {name}"

    def _extract_parameters(self, model_name: str) -> Optional[str]:
        """Extract parameter count from model name.

        Args:
            model_name: Name of the model

        Returns:
            Optional[str]: Parameter count or None if not found
        """
        # Look for common parameter patterns like "7B", "13B", etc.
        match = re.search(r"(\d+\.?\d*)B", model_name)
        if match:
            return match.group(0)
        return None

    def _extract_quantization(self, model_name: str) -> Optional[str]:
        """Extract quantization information from model name.

        Args:
            model_name: Name of the model

        Returns:
            Optional[str]: Quantization info or None if not found
        """
        # Look for common quantization patterns
        patterns = [
            r"Q\d+_\d+",  # Q8_0, Q4_0, etc.
            r"q\d+_\d+",  # q8_0, q4_0, etc.
            r"-q\d+",  # -q8, -q4, etc.
            r"\.q\d+",  # .q8, .q4, etc.
        ]

        for pattern in patterns:
            match = re.search(pattern, model_name)
            if match:
                return match.group(0)

        return None

    def _extract_context_length(self, model_name: str) -> Optional[int]:
        """Extract context length from model name.

        Args:
            model_name: Name of the model

        Returns:
            Optional[int]: Context length in tokens or None if not found
        """
        # Look for patterns like "8k", "32k", etc.
        match = re.search(r"(\d+)k", model_name.lower())
        if match:
            try:
                return int(match.group(1)) * 1000
            except ValueError:
                pass
        return None

    def _extract_tags_from_name(self, model_name: str) -> List[str]:
        """Extract meaningful tags from model name.

        Args:
            model_name: Name of the model

        Returns:
            List[str]: List of extracted tags
        """
        tags = []

        # Common model families
        families = ["llama", "mistral", "stable", "phi", "vicuna", "falcon", "mpt"]
        for family in families:
            if family.lower() in model_name.lower():
                tags.append(family)

        # Extract quantization as a tag
        quant = self._extract_quantization(model_name)
        if quant:
            tags.append(f"quantized-{quant}")

        # Add parameter size as a tag
        params = self._extract_parameters(model_name)
        if params:
            tags.append(f"params-{params}")

        # Add context length as a tag
        ctx_len = self._extract_context_length(model_name)
        if ctx_len:
            tags.append(f"context-{ctx_len//1000}k")

        return tags

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Convert bytes to human-readable size.

        Args:
            size_bytes: Size in bytes

        Returns:
            str: Human-readable size string
        """
        if size_bytes == 0:
            return "0B"

        units = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        size_value = float(size_bytes)

        while size_value >= 1024.0 and i < len(units) - 1:
            size_value /= 1024.0
            i += 1

        # Format with one decimal place for values < 10
        if size_value < 10:
            return f"{size_value:.1f}{units[i]}"
        else:
            return f"{int(size_value)}{units[i]}"


# Convenient functions for direct module usage
def fetch_models() -> List[Dict[str, Any]]:
    """
    Fetches available models from the Ollama API.

    Returns:
        List[Dict[str, Any]]: List of available model data as dictionaries.

    Raises:
        OllamaAPIError: If the API request fails.
    """
    provider = OllamaLocalModelProvider()
    models = provider.list_models()
    return [model.to_dict() for model in models]


# Display Function
def display_models(models: List[Union[Dict[str, Any], ModelInfo]]) -> None:
    """
    Displays model information in a readable format.

    Args:
        models: List of model dictionaries or ModelInfo objects to display.
    """
    if not models:
        print("No models found.")
        return

    print(f"\nFound {len(models)} model(s):")
    for model in models:
        if isinstance(model, ModelInfo):
            name = model.name
            size = str(model.size) if model.size else "N/A"
            digest = model.metadata.get("digest", "N/A") if model.metadata else "N/A"
        else:
            name = model.get("name", "Unknown")
            size = model.get("size", "N/A")
            digest = model.get("digest", "N/A")

        print(f"📌 Model: {name} | Size: {size} | Digest: {digest[:8]}...")


# Main Execution
def main() -> None:
    """
    Main entry point for executing the Ollama model browser.
    """
    print("Welcome to the Eidosian Ollama Model Browser")

    try:
        provider = OllamaLocalModelProvider()
        models = provider.list_models()

        if not models:
            print("No models available on the Ollama server.")
            return

        # Display results
        display_models(models)

        # Optional: Install a model
        install_choice = (
            input("\nWould you like to install a model? (y/n): ").strip().lower()
        )
        if install_choice == "y":
            model_name = input("Enter model name to install: ").strip()

            # Create a dummy model for installation
            model = ModelInfo(name=model_name, source=ModelSource.OLLAMA_LOCAL)

            print(f"Installing {model_name}...")
            success = provider.install_model(model)

            if success:
                print(f"✅ Successfully installed {model_name}")
            else:
                print(f"❌ Failed to install {model_name}")

    except (OllamaAPIError, ValidationError) as e:
        print(f"Error: {e}")

    except KeyboardInterrupt:
        print("\nProcess interrupted. Goodbye.")


if __name__ == "__main__":
    main()
