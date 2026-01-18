"""
Model manager for connecting to and orchestrating LLM providers.

This module provides a unified interface for interacting with various language
model providers through provider-specific adapters. It handles authentication,
request formatting, rate limiting, and response parsing with proper error handling.
"""

import abc
import os
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Final,
    List,
    Literal,
    NotRequired,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    cast,
)

from pydantic import BaseModel, SecretStr, field_validator

from llm_forge.logging_config import configure_logging
from llm_forge.type_definitions import ModelType

# Type checking imports that won't be evaluated at runtime
if TYPE_CHECKING:
    pass

# Configure module-specific logger
logger = configure_logging()

# Type definitions
ProviderType = Literal["openai", "anthropic", "meta", "mistral", "google"]
T = TypeVar("T")


# Common response and parameter types
class TokenUsage(TypedDict):
    """Token usage statistics from model responses."""

    total_tokens: int
    completion_tokens: NotRequired[int]
    prompt_tokens: NotRequired[int]


class LLMResponse(TypedDict):
    """Standardized response format for LLM interactions."""

    content: str
    usage: TokenUsage
    model: str
    finish_reason: NotRequired[str]


class ModelRequestParams(TypedDict):
    """Common parameters for model requests."""

    temperature: NotRequired[float]
    max_tokens: NotRequired[int]
    top_p: NotRequired[float]
    top_k: NotRequired[int]
    stop_sequences: NotRequired[List[str]]
    timeout: NotRequired[float]


class ModelConfig(BaseModel):
    """
    Configuration for a specific language model.

    Attributes:
        provider: The LLM provider (e.g., "openai", "anthropic")
        model_id: The specific model identifier
        api_key: Secret API key for authentication
        api_url: Optional custom API endpoint
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum response length
        timeout: Request timeout in seconds
    """

    provider: ProviderType
    model_id: str
    api_key: SecretStr
    api_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """
        Validate temperature is between 0 and 1.

        Args:
            v: The temperature value to validate

        Returns:
            The validated temperature value

        Raises:
            ValueError: If temperature is outside valid range
        """
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """
        Validate max_tokens is positive.

        Args:
            v: The max_tokens value to validate

        Returns:
            The validated max_tokens value

        Raises:
            ValueError: If max_tokens is not positive
        """
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class LLMProvider(Protocol):
    """Protocol defining the interface for LLM provider implementations."""

    @property
    def provider_name(self) -> ProviderType:
        """Get the provider's name."""
        ...

    @property
    def supported_models(self) -> List[str]:
        """Get list of supported model identifiers."""
        ...

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text using the provider's language model.

        Args:
            prompt: Input text prompt
            **kwargs: Provider-specific parameters

        Returns:
            Generated text response

        Raises:
            ModelRequestError: If the request fails
        """
        ...

    def generate_with_metadata(
        self, prompt: str, **kwargs: Any
    ) -> Tuple[str, Dict[str, float]]:
        """
        Generate text and return with metadata.

        Args:
            prompt: Input text prompt
            **kwargs: Provider-specific parameters

        Returns:
            Tuple of (generated_text, metadata_dict)

        Raises:
            ModelRequestError: If the request fails
        """
        ...


class ModelRequestError(Exception):
    """Exception raised when an LLM request fails."""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        retry_after: Optional[int] = None,
    ) -> None:
        """
        Initialize with error details.

        Args:
            message: Error description
            provider: Name of the LLM provider
            status_code: HTTP status code if applicable
            retry_after: Seconds to wait before retry if rate limited
        """
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(f"{provider} error: {message}")


# Type for OpenAI API errors
class OpenAIError(Exception):
    """Base exception for OpenAI API errors."""

    def __init__(self, *args: Any) -> None:
        """
        Initialize OpenAI error.

        Args:
            *args: Error information
        """
        super().__init__(*args)
        self.response: Optional[Any] = None


class BaseModelProvider(abc.ABC):
    """Abstract base class for LLM provider implementations."""

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize the provider with configuration.

        Args:
            config: Provider and model configuration
        """
        self.config = config
        self._validate_api_key()
        logger.debug(
            f"Initialized {self.provider_name} provider for model {config.model_id}"
        )

    @property
    @abc.abstractmethod
    def provider_name(self) -> ProviderType:
        """Get the provider's name."""
        pass

    @property
    @abc.abstractmethod
    def supported_models(self) -> List[str]:
        """Get list of supported model identifiers."""
        pass

    @abc.abstractmethod
    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        pass

    @abc.abstractmethod
    def generate_with_metadata(
        self, prompt: str, **kwargs: Any
    ) -> Tuple[str, Dict[str, float]]:
        """Generate text with metadata."""
        pass

    def _validate_api_key(self) -> None:
        """
        Validate the API key exists.

        Raises:
            ValueError: If API key is empty
        """
        if not self.config.api_key.get_secret_value():
            raise ValueError(f"API key for {self.provider_name} is required")

    def handle_rate_limit(self, retry_after: Optional[int] = None) -> None:
        """
        Handle rate limiting by waiting.

        Args:
            retry_after: Seconds to wait or None for default
        """
        wait_time = retry_after or 5
        logger.warning(f"{self.provider_name} rate limited. Waiting {wait_time}s")
        time.sleep(wait_time)


class OpenAIProvider(BaseModelProvider):
    """Provider implementation for OpenAI models (GPT series)."""

    @property
    def provider_name(self) -> ProviderType:
        """Get provider name."""
        return "openai"

    @property
    def supported_models(self) -> List[str]:
        """Get supported models."""
        return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text using OpenAI API.

        Args:
            prompt: User input text
            **kwargs: Additional parameters for generation
                temperature: Sampling temperature (0.0-1.0)
                max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            ImportError: If OpenAI package is not installed
            ModelRequestError: If API request fails
        """
        try:
            import openai
            from openai.types.chat import ChatCompletion

            # Configure API key from config
            openai.api_key = self.config.api_key.get_secret_value()

            # Set up request parameters
            params: Dict[str, Any] = {
                "model": self.config.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            }

            # Make the API call
            client = openai.OpenAI(api_key=openai.api_key)
            response: ChatCompletion = client.chat.completions.create(**params)

            # Extract content safely
            content = ""
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content = message.content

            return content

        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with 'pip install openai>=1.0.0'."
            )
        except Exception as e:
            # Handle OpenAI specific errors with proper type checking
            status_code = None
            retry_after = None

            # Check for rate limiting
            if hasattr(e, "response") and getattr(e, "response", None) is not None:
                err_response = getattr(e, "response")
                if hasattr(err_response, "status_code"):
                    status_code = err_response.status_code
                    if status_code == 429:
                        if hasattr(err_response, "headers"):
                            retry_after_header = err_response.headers.get("retry-after")
                            retry_after = (
                                int(retry_after_header) if retry_after_header else 5
                            )
                        raise ModelRequestError(
                            "Rate limit exceeded", "openai", status_code, retry_after
                        )
                    elif status_code >= 500:
                        raise ModelRequestError(str(e), "openai", status_code)

            # For all other errors
            raise ModelRequestError(str(e), "openai", status_code)

    def generate_with_metadata(
        self, prompt: str, **kwargs: Any
    ) -> Tuple[str, Dict[str, float]]:
        """
        Generate text with metadata from OpenAI.

        Args:
            prompt: User input text
            **kwargs: Additional parameters for generation
                temperature: Sampling temperature (0.0-1.0)
                max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_text, metadata_dict) where metadata includes token counts

        Raises:
            ImportError: If OpenAI package is not installed
            ModelRequestError: If API request fails
        """
        try:
            import openai
            from openai.types.chat import ChatCompletion

            # Configure API key from config
            openai.api_key = self.config.api_key.get_secret_value()

            # Set up request parameters
            params: Dict[str, Any] = {
                "model": self.config.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            }

            # Make the API call
            client = openai.OpenAI(api_key=openai.api_key)
            response: ChatCompletion = client.chat.completions.create(**params)

            # Extract text and metadata
            content = ""
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content = message.content

            # Initialize default metadata
            metadata: Dict[str, float] = {
                "total_tokens": 0.0,
                "completion_tokens": 0.0,
                "prompt_tokens": 0.0,
            }

            # Extract token usage metrics if available
            if response.usage:
                if hasattr(response.usage, "total_tokens"):
                    metadata["total_tokens"] = float(response.usage.total_tokens)
                if hasattr(response.usage, "completion_tokens"):
                    metadata["completion_tokens"] = float(
                        response.usage.completion_tokens
                    )
                if hasattr(response.usage, "prompt_tokens"):
                    metadata["prompt_tokens"] = float(response.usage.prompt_tokens)

            return content, metadata

        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with 'pip install openai>=1.0.0'."
            )
        except Exception as e:
            raise ModelRequestError(str(e), "openai")


class AnthropicProvider(BaseModelProvider):
    """Provider implementation for Anthropic models (Claude series)."""

    @property
    def provider_name(self) -> ProviderType:
        """Get provider name."""
        return "anthropic"

    @property
    def supported_models(self) -> List[str]:
        """Get supported models."""
        return ["claude-2", "claude-instant-1"]

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text using Anthropic API.

        Args:
            prompt: User input text
            **kwargs: Additional parameters for generation
                temperature: Sampling temperature (0.0-1.0)
                max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            ImportError: If Anthropic package is not installed
            ModelRequestError: If API request fails
        """
        try:
            import anthropic
            from anthropic.types import Completion

            # Initialize client
            client = anthropic.Anthropic(api_key=self.config.api_key.get_secret_value())

            # Set up request parameters
            params: Dict[str, Any] = {
                "model": self.config.model_id,
                "prompt": f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens_to_sample": kwargs.get(
                    "max_tokens", self.config.max_tokens
                ),
            }

            # Make the API call
            response: Completion = client.completions.create(**params)

            # Extract content safely
            completion = ""
            if hasattr(response, "completion"):
                completion = str(response.completion)

            return completion

        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with 'pip install anthropic'."
            )
        except Exception as e:
            if "429" in str(e):
                raise ModelRequestError("Rate limit exceeded", "anthropic", 429)
            raise ModelRequestError(str(e), "anthropic")

    def generate_with_metadata(
        self, prompt: str, **kwargs: Any
    ) -> Tuple[str, Dict[str, float]]:
        """
        Generate text with metadata from Anthropic.

        Args:
            prompt: User input text
            **kwargs: Additional parameters for generation
                temperature: Sampling temperature (0.0-1.0)
                max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_text, metadata_dict) where metadata includes estimated token count

        Raises:
            ImportError: If Anthropic package is not installed
            ModelRequestError: If API request fails
        """
        # Get the text response
        text = self.generate_text(prompt, **kwargs)

        # Anthropic doesn't return token counts in the same way
        # Estimate based on text length (rough approximation)
        metadata: Dict[str, float] = {
            "estimated_tokens": float(len(text) / 4),  # Very rough estimate
        }

        return text, metadata


class MistralProvider(BaseModelProvider):
    """Provider implementation for Mistral AI models."""

    @property
    def provider_name(self) -> ProviderType:
        """Get provider name."""
        return "mistral"

    @property
    def supported_models(self) -> List[str]:
        """Get supported models."""
        return ["mistral-small", "mistral-medium", "mistral-large"]

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text using Mistral AI API.

        Args:
            prompt: User input text
            **kwargs: Additional parameters for generation
                temperature: Sampling temperature (0.0-1.0)
                max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            ImportError: If MistralAI package is not installed
            ModelRequestError: If API request fails
        """
        try:
            # Import locally to handle missing dependency gracefully
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import (
                ChatCompletionResponse,
                ChatMessage,
            )

            # Initialize client
            client = MistralClient(api_key=self.config.api_key.get_secret_value())

            # Create message list with proper typing
            messages = [ChatMessage(role="user", content=prompt)]

            # Make the API call
            response: ChatCompletionResponse = client.chat(
                model=self.config.model_id,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )

            # Extract content safely
            content = ""
            if (
                hasattr(response, "choices")
                and response.choices
                and len(response.choices) > 0
            ):
                message = response.choices[0].message
                if hasattr(message, "content") and message.content:
                    content = message.content

            return content

        except ImportError:
            raise ImportError(
                "Mistral AI package not installed. Install with 'pip install mistralai'."
            )
        except Exception as e:
            raise ModelRequestError(str(e), "mistral")

    def generate_with_metadata(
        self, prompt: str, **kwargs: Any
    ) -> Tuple[str, Dict[str, float]]:
        """
        Generate text with metadata from Mistral.

        Args:
            prompt: User input text
            **kwargs: Additional parameters for generation
                temperature: Sampling temperature (0.0-1.0)
                max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_text, metadata_dict) where metadata includes token usage

        Raises:
            ImportError: If MistralAI package is not installed
            ModelRequestError: If API request fails
        """
        try:
            # Import locally to handle missing dependency gracefully
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import (
                ChatCompletionResponse,
                ChatMessage,
            )

            # Initialize client
            client = MistralClient(api_key=self.config.api_key.get_secret_value())

            # Create message list with proper typing
            messages = [ChatMessage(role="user", content=prompt)]

            # Make the API call
            response: ChatCompletionResponse = client.chat(
                model=self.config.model_id,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )

            # Extract content safely
            content = ""
            if (
                hasattr(response, "choices")
                and response.choices
                and len(response.choices) > 0
            ):
                message = response.choices[0].message
                if hasattr(message, "content") and message.content:
                    content = message.content

            # Extract metadata safely
            metadata: Dict[str, float] = {
                "usage": 0.0,
            }

            if hasattr(response, "usage") and response.usage:
                if hasattr(response.usage, "total_tokens"):
                    metadata["usage"] = float(response.usage.total_tokens)

            return content, metadata

        except ImportError:
            raise ImportError(
                "Mistral AI package not installed. Install with 'pip install mistralai'."
            )
        except Exception as e:
            raise ModelRequestError(str(e), "mistral")


class ModelManager:
    """
    Central manager for LLM model providers and interactions.

    This class manages connections to different LLM providers,
    handles provider selection, and provides a unified interface
    for text generation.
    """

    # Map model types to provider classes
    _PROVIDER_MAP: Final[Dict[ProviderType, Type[BaseModelProvider]]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "mistral": MistralProvider,
    }

    # Map model identifiers to provider types
    _MODEL_TO_PROVIDER: Final[Dict[str, ProviderType]] = {
        "gpt": "openai",
        "claude": "anthropic",
        "mistral": "mistral",
    }

    def __init__(self) -> None:
        """Initialize the model manager."""
        self._providers: Dict[str, BaseModelProvider] = {}
        self._load_environment_config()
        logger.info("Initialized ModelManager")

    def _load_environment_config(self) -> None:
        """Load configuration from environment variables."""
        # Check for API keys in environment
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        mistral_key = os.environ.get("MISTRAL_API_KEY")

        # Configure providers with available keys
        if openai_key:
            self.configure_provider("openai", "gpt-3.5-turbo", openai_key)

        if anthropic_key:
            self.configure_provider("anthropic", "claude-2", anthropic_key)

        if mistral_key:
            self.configure_provider("mistral", "mistral-medium", mistral_key)

    def configure_provider(
        self, provider: ProviderType, model_id: str, api_key: str, **kwargs: Any
    ) -> None:
        """
        Configure and initialize a provider.

        Args:
            provider: Provider type identifier
            model_id: Model identifier
            api_key: API key for authentication
            **kwargs: Additional configuration parameters

        Raises:
            ValueError: If provider type is invalid
        """
        if provider not in self._PROVIDER_MAP:
            raise ValueError(f"Unsupported provider: {provider}")

        # Create config
        config = ModelConfig(
            provider=provider, model_id=model_id, api_key=SecretStr(api_key), **kwargs
        )

        # Initialize provider
        provider_class = self._PROVIDER_MAP[provider]
        provider_instance = provider_class(config)

        # Store provider with unique key
        provider_key = f"{provider}:{model_id}"
        self._providers[provider_key] = provider_instance
        logger.info(f"Configured provider: {provider_key}")

    def get_provider_for_model(
        self, model_type: ModelType
    ) -> Optional[BaseModelProvider]:
        """
        Get the appropriate provider for a model type.

        Args:
            model_type: Type of model to use

        Returns:
            Provider instance or None if not configured
        """
        provider_type = self._MODEL_TO_PROVIDER.get(str(model_type))
        if not provider_type:
            logger.warning(f"No provider mapping for model type: {model_type}")
            return None

        # Find a configured provider that matches
        for _, provider in self._providers.items():
            if provider.provider_name == provider_type:
                return provider

        logger.warning(f"No configured provider for model type: {model_type}")
        return None

    def generate_text(
        self, prompt: str, model: ModelType, max_retries: int = 3, **kwargs: Any
    ) -> str:
        """
        Generate text using the specified model type.

        Args:
            prompt: Input text prompt
            model: Type of model to use
            max_retries: Maximum retry attempts for rate limits
            **kwargs: Additional generation parameters

        Returns:
            Generated text response

        Raises:
            ValueError: If no provider is configured for the model
            ModelRequestError: If generation fails after retries
        """
        provider = self.get_provider_for_model(model)
        if not provider:
            raise ValueError(f"No provider configured for model: {model}")

        # Attempt generation with retries
        attempts = 0
        last_error: Optional[ModelRequestError] = None

        while attempts < max_retries:
            try:
                return provider.generate_text(prompt, **kwargs)
            except ModelRequestError as e:
                last_error = e
                attempts += 1

                # Handle rate limiting specially
                if e.status_code == 429 and e.retry_after:
                    provider.handle_rate_limit(e.retry_after)
                elif attempts < max_retries:
                    # Exponential backoff for other errors
                    wait_time = 2**attempts
                    logger.warning(f"Retrying after error: {e} (waiting {wait_time}s)")
                    time.sleep(wait_time)

        # If we get here, all retries failed
        if last_error:
            raise last_error
        raise ModelRequestError("Generation failed", str(model))

    def get_available_models(self) -> Dict[ModelType, List[str]]:
        """
        Get all available configured models.

        Returns:
            Dictionary mapping model types to available model IDs
        """
        available: Dict[ModelType, List[str]] = {}

        for _, provider in self._providers.items():
            provider_name = provider.provider_name

            # Map provider to model type
            for model_type, provider_type in self._MODEL_TO_PROVIDER.items():
                if provider_type == provider_name:
                    model_type_cast = cast(ModelType, model_type)
                    if model_type_cast not in available:
                        available[model_type_cast] = []
                    available[model_type_cast].append(provider.config.model_id)

        return available


# Singleton instance for global access
_instance: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    Get the global ModelManager instance.

    Returns:
        Singleton ModelManager instance
    """
    global _instance
    if _instance is None:
        _instance = ModelManager()
    return _instance
