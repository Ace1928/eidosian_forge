"""
Eidosian Model Management System
================================

Universal abstractions for multi-source model management with rigorous
type safety and elegant consistency across heterogeneous model sources.

Core interfaces and protocols ensure complete compatibility between
Ollama, Hugging Face, and potentially additional future model sources.
"""

from ollama_forge.src.ollama_forge.models.browse_remote_huggingface_models import (
    HuggingFaceAPIError,
    ValidationError,
)
from ollama_forge.src.ollama_forge.models.browse_remote_huggingface_models import (
    display_models as display_remote_huggingface_models,
)
from ollama_forge.src.ollama_forge.models.browse_remote_huggingface_models import (
    fetch_models as fetch_remote_huggingface_models,
)
from ollama_forge.src.ollama_forge.models.browse_remote_huggingface_models import (
    filter_models as filter_remote_huggingface_models,
)
from ollama_forge.src.ollama_forge.models.browse_remote_huggingface_models import (
    sort_models as sort_remote_huggingface_models,
)
from ollama_forge.src.ollama_forge.models.browse_remote_ollama_models import (
    DetailedModelExtractor,
    ModelIndexer,
    ModelSearcher,
    OllamaLibraryScraper,
)
from ollama_forge.src.ollama_forge.models.browse_remote_ollama_models import (
    display_models as display_remote_ollama_models,
)
from ollama_forge.src.ollama_forge.models.list_local_ollama_models import (
    OllamaLocalModelProvider,
)
from ollama_forge.src.ollama_forge.models.list_local_ollama_models import (
    display_models as display_local_ollama_models,
)
from ollama_forge.src.ollama_forge.models.list_local_ollama_models import (
    fetch_models as fetch_local_ollama_models,
)
from ollama_forge.src.ollama_forge.models.model_manager import (
    HuggingFaceModelProvider,
    ModelManager,
    OllamaRemoteModelProvider,
    create_cli,
)
from ollama_forge.src.ollama_forge.utils.type_definitions import (
    DownloadProgress,
    ModelData,
    ModelInfo,
    ModelInstaller,
    ModelProvider,
    ModelSize,
    ModelSource,
)

# Export main abstractions
__all__ = [
    "ModelData",
    "ModelSource",
    "ModelSize",
    "ModelInfo",
    "ModelProvider",
    "ModelInstaller",
    "DownloadProgress",
    "HuggingFaceAPIError",
    "ValidationError",
    "ModelSearcher",
    "ModelIndexer",
    "DetailedModelExtractor",
    "OllamaLibraryScraper",
    "OllamaLocalModelProvider",
    "OllamaRemoteModelProvider",
    "HuggingFaceModelProvider",
    "ModelManager",
    "create_cli",
    "OllamaLibraryScraper",
    "ModelSearcher",
    "ModelIndexer",
    "DetailedModelExtractor",
    "OllamaLibraryScraper",
    "OllamaLocalModelProvider",
    "OllamaRemoteModelProvider",
    "HuggingFaceModelProvider",
    "ModelManager",
    "create_cli",
    "display_remote_ollama_models",
    "display_local_ollama_models",
    "fetch_local_ollama_models",
    "display_remote_huggingface_models",
    "fetch_remote_huggingface_models",
    "filter_remote_huggingface_models",
    "sort_remote_huggingface_models",
]
