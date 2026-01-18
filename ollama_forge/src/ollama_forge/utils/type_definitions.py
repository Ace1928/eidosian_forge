from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, TypedDict, TypeVar


# Type definitions for API responses
class MessageContent(TypedDict):
    """Type for message content in chat responses."""

    role: str
    content: str


class GenerateResponse(TypedDict):
    """Type for generation response."""

    response: str


class ChatResponse(TypedDict):
    """Type for chat response."""

    message: MessageContent


class EmbeddingResponse(TypedDict):
    """Type for embedding response."""

    embedding: List[float]


# Type definitions
ModelData = Dict[str, Any]
T = TypeVar("T")


class ModelSource(Enum):
    """Sources from which models can be retrieved or managed."""

    OLLAMA_LOCAL = auto()
    OLLAMA_REMOTE = auto()
    HUGGINGFACE = auto()


@dataclass
class ModelSize:
    """Structured representation of model size information."""

    value: float  # Size value
    unit: str  # GB, MB, etc.

    def __str__(self) -> str:
        """String representation of size."""
        return f"{self.value}{self.unit}"

    @classmethod
    def parse(cls, size_str: Optional[str]) -> Optional["ModelSize"]:
        """Parse a size string into a structured object.

        Args:
            size_str: Size string like "13B" or "4.7GB"

        Returns:
            Optional[ModelSize]: Structured size or None if parsing fails
        """
        import re

        if not size_str:
            return None

        # Extract numeric value and unit
        match = re.search(r"([\d.]+)\s*([A-Za-z]+)", size_str)
        if not match:
            return None

        try:
            value = float(match.group(1))
            unit = match.group(2)
            return cls(value=value, unit=unit)
        except (ValueError, IndexError):
            return None


class DownloadProgress(Protocol):
    """Protocol for reporting download progress."""

    def update(self, bytes_downloaded: int, total_bytes: int) -> None:
        """Update progress with downloaded bytes.

        Args:
            bytes_downloaded: Number of bytes downloaded so far
            total_bytes: Total bytes to download (may be 0 if unknown)
        """
        ...


@dataclass
class ModelInfo:
    """Universal model information representation across sources."""

    name: str
    source: ModelSource
    description: str = ""
    tags: List[str] = field(default_factory=list)
    size: Optional[ModelSize] = None
    parameters: Optional[str] = None
    context_length: Optional[int] = None
    url: Optional[str] = None
    quantization: Optional[str] = None
    installed: bool = False
    # Extension point for source-specific data
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "name": self.name,
            "source": self.source.name,
            "description": self.description,
            "tags": self.tags,
            "size": str(self.size) if self.size else None,
            "parameters": self.parameters,
            "context_length": self.context_length,
            "url": self.url,
            "quantization": self.quantization,
            "installed": self.installed,
            "metadata": self.metadata,
        }
        return result


class ModelProvider(Protocol):
    """Protocol defining the interface for model source providers."""

    def list_models(
        self, query: Optional[str] = None, installed_only: bool = False, **kwargs: Any
    ) -> List[ModelInfo]:
        """List available models from this source.

        Args:
            query: Optional search query
            installed_only: If True, only return installed models
            **kwargs: Additional source-specific parameters

        Returns:
            List[ModelInfo]: Available models matching criteria
        """
        ...

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model.

        Args:
            name: Name of the model to retrieve

        Returns:
            Optional[ModelInfo]: Model information or None if not found
        """
        ...


class ModelInstaller(Protocol):
    """Protocol for model installation capabilities."""

    def install_model(
        self, model: ModelInfo, progress_callback: Optional[DownloadProgress] = None
    ) -> bool:
        """Install a model from its source.

        Args:
            model: Model to install
            progress_callback: Optional callback for reporting installation progress

        Returns:
            bool: True if installation was successful
        """
        ...

    def uninstall_model(self, model_name: str) -> bool:
        """Uninstall a previously installed model.

        Args:
            model_name: Name of the model to uninstall

        Returns:
            bool: True if uninstallation was successful
        """
        ...


class ModelsResponse(TypedDict):
    """Type for models list response."""

    models: List[ModelInfo]


class VersionResponse(TypedDict):
    """Type for version response."""

    version: str


class PullProgress(TypedDict, total=False):
    """Type for model pull progress information."""

    completed: int
    total: int  # Optional due to total=False
    status: str
