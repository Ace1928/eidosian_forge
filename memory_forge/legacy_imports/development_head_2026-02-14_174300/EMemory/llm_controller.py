#!/usr/bin/env python3
"""
🔹 The Ultimate Eidosian LLM Controller Module 🔹

Implements a modular and extensible interface for interacting with
Large Language Models (LLMs). Focuses on local execution with Ollama.

Features:
    • OllamaLLMController (default: "deepseek-r1:1.5b")
    • ChatOllamaLLMController (for plain text / chat usage)
    • OpenAILLMController  (cloud: "gpt-4")
    • Automatic retries for calls to Ollama
    • Response caching to avoid redundant calls
    • Centralized configuration for model parameters
"""

import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import requests
from eidos_config import BACKOFF_SECONDS, DEFAULT_OLLLAMA_API_BASE, DEFAULT_TEMPERATURE, MAX_RETRIES, MAX_TOKENS
from litellm import completion
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# =============================================================================
# Constants & Exceptions
# =============================================================================
SYSTEM_PROMPT: str = "You must respond with a JSON object."


class LLMControllerError(Exception):
    """Base exception class for LLM controller errors."""

    pass


class ModelNotAvailableError(LLMControllerError):
    """Exception raised when a requested model is not available."""

    pass


class InvalidParameterError(LLMControllerError):
    """Exception raised when invalid parameters are provided."""

    pass


class ConnectionError(LLMControllerError):
    """Exception raised when connection to the LLM service fails."""

    pass


class ResponseParseError(LLMControllerError):
    """Exception raised when the LLM response cannot be parsed."""

    pass


class ModelOverloadedError(LLMControllerError):
    """Exception raised when the LLM service is overloaded."""

    pass


# =============================================================================
# Utility Functions
# =============================================================================
def build_default_messages(prompt: str) -> List[Dict[str, str]]:
    """
    Constructs the default messages list for JSON-based usage.
    Enforces "You must respond with a JSON object." in the system role.

    Args:
        prompt (str): The user prompt to include.

    Returns:
        List[Dict[str, str]]: A list of message dictionaries with system and user roles.
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def build_chat_messages(prompt: str) -> List[Dict[str, str]]:
    """
    A chat-friendly approach that uses a simpler system role,
    suitable for plain text or minimal constraints.

    Args:
        prompt (str): The user prompt to include.

    Returns:
        List[Dict[str, str]]: A list of message dictionaries with system and user roles.
    """
    return [
        {"role": "system", "content": "You are a helpful AI. Respond as plain text."},
        {"role": "user", "content": prompt},
    ]


def sanitize_prompt(prompt: str) -> str:
    """
    Sanitizes the input prompt to remove potentially problematic elements.

    Args:
        prompt (str): The input prompt to sanitize.

    Returns:
        str: The sanitized prompt.
    """
    if not prompt:
        return ""

    # Remove any control characters
    sanitized = re.sub(r"[\x00-\x1F\x7F]", "", prompt)

    # Limit length to prevent too large prompts
    max_length = 32000  # Adjust as needed
    if len(sanitized) > max_length:
        logger.warning(f"Prompt truncated from {len(sanitized)} to {max_length} characters")
        sanitized = sanitized[:max_length]

    return sanitized


def generate_cache_key(prompt: str, response_format: dict, temperature: float, model: str) -> str:
    """
    Generates a unique cache key for a set of LLM call parameters.

    Args:
        prompt (str): The prompt text.
        response_format (dict): The response format specification.
        temperature (float): The sampling temperature.
        model (str): The model identifier.

    Returns:
        str: A unique hash representing this parameter combination.
    """
    # Create a string representation of all parameters
    params_str = f"{prompt}|{json.dumps(response_format)}|{temperature}|{model}"

    # Generate an MD5 hash (sufficient for caching purposes)
    return hashlib.md5(params_str.encode("utf-8")).hexdigest()


# =============================================================================
# Response Cache Implementation
# =============================================================================
@dataclass
class CacheItem:
    """Represents a cached LLM response."""

    response: str
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0


class ResponseCache:
    """
    Manages caching of LLM responses to avoid redundant API calls.
    Implements a simple time-based expiration strategy.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Initialize the response cache.

        Args:
            max_size (int): Maximum number of items to keep in cache.
            ttl_seconds (int): Time-to-live in seconds for cache items.
        """
        self.cache: Dict[str, CacheItem] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[str]:
        """
        Retrieve a cached response if available and not expired.

        Args:
            key (str): The cache key to look up.

        Returns:
            Optional[str]: The cached response or None if not found/expired.
        """
        if key not in self.cache:
            return None

        item = self.cache[key]
        current_time = time.time()

        # Check if item is expired
        if current_time - item.timestamp > self.ttl_seconds:
            del self.cache[key]
            return None

        # Update access count and return response
        item.access_count += 1
        return item.response

    def put(self, key: str, response: str) -> None:
        """
        Store a response in the cache.

        Args:
            key (str): The cache key.
            response (str): The response to cache.
        """
        # If cache is full, remove least accessed item
        if len(self.cache) >= self.max_size:
            least_used_key = min(self.cache.items(), key=lambda x: x[1].access_count)[0]
            del self.cache[least_used_key]

        # Store new item
        self.cache[key] = CacheItem(response=response)

    def clear(self) -> None:
        """Clear all items from the cache."""
        self.cache.clear()


# Global response cache instance
response_cache = ResponseCache()


# =============================================================================
# Abstract Base
# =============================================================================
class AbstractLLMController(ABC):
    """
    Abstract base class for all LLM controllers.
    Enforces the implementation of required methods for synchronous,
    asynchronous, and streaming operations.
    """

    @abstractmethod
    def get_completion(self, prompt: str, response_format: dict, temperature: float) -> str:
        """
        Generates a response from the configured LLM backend.

        Args:
            prompt (str): The input query.
            response_format (dict): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The LLM response as a JSON-formatted or plain string.

        Raises:
            LLMControllerError: Base class for all controller errors.
            ModelNotAvailableError: When the requested model is not available.
            ConnectionError: When connection to the LLM service fails.
            ResponseParseError: When the response cannot be parsed.
        """
        pass

    async def get_completion_async(self, prompt: str, response_format: dict, temperature: float) -> str:
        """
        Asynchronous version of get_completion. Default implementation runs the synchronous
        method in a thread pool to maintain compatibility with existing controllers.

        Args:
            prompt (str): The input query.
            response_format (dict): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The LLM response
        """
        import asyncio

        return await asyncio.to_thread(self.get_completion, prompt, response_format, temperature)

    def get_completion_stream(self, prompt: str, response_format: dict, temperature: float) -> str:
        """
        Generates a streaming response from the LLM backend.
        Default implementation uses non-streaming method for backward compatibility.

        Args:
            prompt (str): The input query.
            response_format (dict): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The complete response
        """
        # Default implementation falls back to regular completion
        return self.get_completion(prompt, response_format, temperature)

    async def get_completion_stream_async(self, prompt: str, response_format: dict, temperature: float) -> str:
        """
        Asynchronous version of streaming completion.
        Default implementation runs the synchronous method in a thread pool.

        Args:
            prompt (str): The input query.
            response_format (dict): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The complete response
        """
        import asyncio

        return await asyncio.to_thread(self.get_completion_stream, prompt, response_format, temperature)

    def check_health(self) -> Tuple[bool, str]:
        """
        Performs a health check on the LLM service.

        Returns:
            Tuple[bool, str]: A tuple containing (is_healthy, status_message)
        """
        try:
            # Simple prompt to test if the model is responding
            response = self.get_completion("Respond with the word 'healthy'.", response_format={}, temperature=0.0)
            return True, "Model is responding correctly"
        except Exception as e:
            return False, f"Health check failed: {str(e)}"


# =============================================================================
# Ollama LLM (JSON-based)
# =============================================================================
class OllamaLLMController(AbstractLLMController):
    """
    LLM controller for the Ollama local backend, expecting JSON-based usage by default.

    Uses "deepseek-r1:1.5b" as the default model. Adds retry logic to handle
    transient errors when calling the local server.
    """

    DEFAULT_MODEL: str = "deepseek-r1:1.5b"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        """
        Initialize the Ollama LLM controller.

        Args:
            model (str): The Ollama model to use. Defaults to "deepseek-r1:1.5b".

        Raises:
            ImportError: If the ollama package is not installed.
        """
        try:
            import ollama
        except ImportError:
            raise ImportError("Ollama package not installed. Please install with 'pip install ollama'.")

        self.model: str = model
        self.api_base: str = DEFAULT_OLLLAMA_API_BASE
        self._check_model_availability()
        logger.info(f"OllamaLLMController initialized with model '{model}' at {self.api_base}")

    def _check_model_availability(self) -> bool:
        """
        Checks if the configured model is available in Ollama.

        Returns:
            bool: True if the model is available.

        Raises:
            ModelNotAvailableError: If the model is not available.
        """
        try:
            # Make a lightweight call to the Ollama API to check model availability
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()

            # Check if our model is in the list of available models
            available_models = [model["name"] for model in data.get("models", [])]
            if self.model not in available_models:
                logger.warning(f"Model '{self.model}' not found in available models: {available_models}")
                # Continue execution as the model may be available but not listed
                return False

            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check model availability: {e}")
            # We'll continue execution rather than raising an exception here,
            # as the model might still work even if we can't check availability
            return False

    @staticmethod
    def _generate_empty_value(schema_type: str) -> Any:
        """
        Generates an empty value based on the JSON schema type.

        Args:
            schema_type (str): The JSON schema type.

        Returns:
            Any: An empty value matching the schema type.
        """
        return {"array": [], "string": "", "object": {}, "number": 0, "boolean": False, "null": None}.get(
            schema_type, None
        )

    def _generate_empty_response(self, response_format: dict) -> dict:
        """
        Generates an empty response matching the provided JSON schema.
        Used as a fallback when the LLM call fails.

        Args:
            response_format (dict): The JSON schema specification.

        Returns:
            dict: An empty response matching the schema.
        """
        schema: dict = response_format.get("json_schema", {}).get("schema", {})
        props = schema.get("properties", {})

        if not props:
            # If no properties are specified, return an empty object
            return {}

        return {
            prop: self._generate_empty_value(prop_schema.get("type", "string")) for prop, prop_schema in props.items()
        }

    def _should_use_cache(self, prompt: str, response_format: dict, temperature: float) -> bool:
        """
        Determines if caching should be used for this request.

        Args:
            prompt (str): The input prompt.
            response_format (dict): The response format specification.
            temperature (float): The sampling temperature.

        Returns:
            bool: True if caching should be used.
        """
        # Don't cache requests with high temperature (non-deterministic)
        if temperature > 0.1:
            return False

        # Don't cache very short prompts
        if len(prompt) < 10:
            return False

        return True

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=BACKOFF_SECONDS, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
        reraise=True,
    )
    def _call_ollama(self, prompt: str, response_format: dict, temperature: float) -> str:
        """
        Internal method to call the Ollama API with retry logic.

        Args:
            prompt (str): The input prompt.
            response_format (dict): The response format specification.
            temperature (float): The sampling temperature.

        Returns:
            str: The LLM response.

        Raises:
            ConnectionError: When connection to Ollama fails.
            ResponseParseError: When the response cannot be parsed.
        """
        try:
            # Check if we have a cached response
            if self._should_use_cache(prompt, response_format, temperature):
                cache_key = generate_cache_key(prompt, response_format, temperature, self.model)
                cached_response = response_cache.get(cache_key)

                if cached_response:
                    logger.debug("Retrieved response from cache")
                    return cached_response

            # No cached response, call the API
            start_time = time.time()
            logger.debug(f"Calling Ollama with prompt length: {len(prompt)}")

            # Avoid passing response_format if it's empty to prevent KeyError: 'type'
            kwargs = {
                "model": f"ollama_chat/{self.model}",
                "messages": build_default_messages(prompt),
                "api_base": self.api_base,
                "temperature": temperature,
                "max_tokens": MAX_TOKENS,
                "timeout": 60,  # Add timeout to prevent hanging
            }

            # Only add response_format if it has content
            if response_format and "json_schema" in response_format:
                kwargs["response_format"] = response_format

            response = completion(**kwargs)

            result = response.choices[0].message.content
            elapsed_time = time.time() - start_time
            logger.debug(f"Ollama response received in {elapsed_time:.2f}s")

            # Cache the response if appropriate
            if self._should_use_cache(prompt, response_format, temperature):
                cache_key = generate_cache_key(prompt, response_format, temperature, self.model)
                response_cache.put(cache_key, result)

            return result

        except requests.exceptions.Timeout:
            logger.error("Timeout calling Ollama API")
            raise ConnectionError("Timeout while calling Ollama API")

        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error with Ollama at {self.api_base}")
            raise ConnectionError(f"Failed to connect to Ollama at {self.api_base}")

        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            raise ResponseParseError(f"Failed to parse Ollama response: {str(e)}")

    def get_completion(self, prompt: str, response_format: dict, temperature: float = DEFAULT_TEMPERATURE) -> str:
        """
        Generates a response from the Ollama API.

        Args:
            prompt (str): The input query.
            response_format (dict): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The LLM response.

        Raises:
            Various exceptions wrapped in LLMControllerError subclasses.
        """
        sanitized_prompt = sanitize_prompt(prompt)

        try:
            return self._call_ollama(sanitized_prompt, response_format, temperature)
        except Exception as e:
            logger.error(f"OllamaLLMController Error: {e}")
            if response_format:
                # Return a valid but empty JSON response
                return json.dumps(self._generate_empty_response(response_format))
            return ""

    def get_completion_stream(
        self, prompt: str, response_format: dict, temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """
        Generates a streaming response from the Ollama API.
        Currently simulates streaming by returning the complete response.

        Args:
            prompt (str): The input query.
            response_format (dict): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The complete response
        """
        # Currently Ollama via litellm doesn't have a clean streaming implementation
        # For now, we'll just use the regular completion method
        logger.debug("Streaming requested, but falling back to regular completion for Ollama")
        return self.get_completion(prompt, response_format, temperature)


# =============================================================================
# Ollama LLM (Plain Text / Chat Mode)
# =============================================================================
class ChatOllamaLLMController(AbstractLLMController):
    """
    Variation of Ollama local backend for chat usage (plain text).
    Does NOT force "You must respond with a JSON object." in the system message.
    """

    DEFAULT_MODEL: str = "deepseek-r1:1.5b"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        """
        Initialize the Chat Ollama LLM controller.

        Args:
            model (str): The Ollama model to use. Defaults to "deepseek-r1:1.5b".

        Raises:
            ImportError: If the ollama package is not installed.
        """
        try:
            import ollama
        except ImportError:
            raise ImportError("Ollama package not installed. Please install with 'pip install ollama'.")

        self.model: str = model
        self.api_base: str = DEFAULT_OLLLAMA_API_BASE
        logger.info(f"ChatOllamaLLMController initialized with model '{model}' at {self.api_base}")

    def _should_use_cache(self, prompt: str, temperature: float) -> bool:
        """
        Determines if caching should be used for this request.

        Args:
            prompt (str): The input prompt.
            temperature (float): The sampling temperature.

        Returns:
            bool: True if caching should be used.
        """
        # Don't cache requests with high temperature (non-deterministic)
        if temperature > 0.1:
            return False

        # Don't cache very short prompts
        if len(prompt) < 10:
            return False

        return True

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=BACKOFF_SECONDS, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
        reraise=True,
    )
    def _call_ollama(self, prompt: str, temperature: float) -> str:
        """
        Internal method to call the Ollama API with retry logic for chat mode.

        Args:
            prompt (str): The input prompt.
            temperature (float): The sampling temperature.

        Returns:
            str: The LLM response.

        Raises:
            ConnectionError: When connection to Ollama fails.
            ResponseParseError: When the response cannot be parsed.
        """
        try:
            # Check if we have a cached response
            if self._should_use_cache(prompt, temperature):
                cache_key = generate_cache_key(prompt, {}, temperature, self.model)
                cached_response = response_cache.get(cache_key)

                if cached_response:
                    logger.debug("Retrieved response from cache")
                    return cached_response

            # No cached response, call the API
            start_time = time.time()
            logger.debug(f"Calling Ollama (chat mode) with prompt length: {len(prompt)}")

            response = completion(
                model=f"ollama_chat/{self.model}",
                messages=build_chat_messages(prompt),
                api_base=self.api_base,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                timeout=60,  # Add timeout to prevent hanging
            )

            result = response.choices[0].message.content
            elapsed_time = time.time() - start_time
            logger.debug(f"Ollama chat response received in {elapsed_time:.2f}s")

            # Cache the response if appropriate
            if self._should_use_cache(prompt, temperature):
                cache_key = generate_cache_key(prompt, {}, temperature, self.model)
                response_cache.put(cache_key, result)

            return result

        except requests.exceptions.Timeout:
            logger.error("Timeout calling Ollama API")
            raise ConnectionError("Timeout while calling Ollama API")

        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error with Ollama at {self.api_base}")
            raise ConnectionError(f"Failed to connect to Ollama at {self.api_base}")

        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            raise ResponseParseError(f"Failed to parse Ollama response: {str(e)}")

    def get_completion(self, prompt: str, response_format: dict, temperature: float = DEFAULT_TEMPERATURE) -> str:
        """
        Generates a response from the Ollama API in chat mode.
        Note: response_format is ignored in chat mode.

        Args:
            prompt (str): The input query.
            response_format (dict): Ignored in chat mode.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The LLM plain text response.

        Raises:
            Various exceptions wrapped in LLMControllerError subclasses.
        """
        sanitized_prompt = sanitize_prompt(prompt)

        try:
            return self._call_ollama(sanitized_prompt, temperature)
        except Exception as e:
            logger.error(f"ChatOllamaLLMController Error: {e}")
            # Return empty string on error
            return ""

    def get_completion_stream(
        self, prompt: str, response_format: dict, temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """
        Generates a streaming response from the Ollama API in chat mode.
        Currently simulates streaming by returning the complete response.

        Args:
            prompt (str): The input query.
            response_format (dict): JSON schema or empty dict if none is needed (ignored in chat mode).
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The complete response
        """
        # Currently Ollama via litellm doesn't have a clean streaming implementation
        # For now, we'll just use the regular completion method
        logger.debug("Streaming requested, but falling back to regular completion for ChatOllama")
        return self.get_completion(prompt, response_format, temperature)


# =============================================================================
# OpenAI LLM
# =============================================================================
class OpenAILLMController(AbstractLLMController):
    """
    LLM controller for OpenAI's API.
    Requires a valid API key set via argument or environment variable.
    """

    DEFAULT_MODEL: str = "gpt-4"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None) -> None:
        """
        Initialize the OpenAI LLM controller.

        Args:
            model (str): The OpenAI model to use. Defaults to "gpt-4".
            api_key (Optional[str]): OpenAI API key. If None, uses OPENAI_API_KEY env var.

        Raises:
            ImportError: If the openai package is not installed.
            ValueError: If no API key is provided or found in environment.
        """
        try:
            from openai import OpenAI

            self.model: str = model
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

            if not self.client.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass it.")

            logger.info(f"OpenAILLMController initialized with model '{model}'")

        except ImportError as ie:
            raise ImportError("OpenAI package not found. Install it with: pip install openai") from ie

    def _should_use_cache(self, prompt: str, response_format: dict, temperature: float) -> bool:
        """
        Determines if caching should be used for this request.

        Args:
            prompt (str): The input prompt.
            response_format (dict): The response format specification.
            temperature (float): The sampling temperature.

        Returns:
            bool: True if caching should be used.
        """
        # Only cache deterministic requests
        if temperature > 0.0:
            return False

        # Only cache if prompt is substantial
        if len(prompt) < 20:
            return False

        return True

    def get_completion(self, prompt: str, response_format: dict, temperature: float = DEFAULT_TEMPERATURE) -> str:
        """
        Generates a response from the OpenAI API.

        Args:
            prompt (str): The input query.
            response_format (dict): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The LLM response.

        Raises:
            Various exceptions wrapped in LLMControllerError subclasses.
        """
        sanitized_prompt = sanitize_prompt(prompt)

        # Check cache first
        if self._should_use_cache(sanitized_prompt, response_format, temperature):
            cache_key = generate_cache_key(sanitized_prompt, response_format, temperature, self.model)
            cached_response = response_cache.get(cache_key)

            if cached_response:
                logger.debug("Retrieved OpenAI response from cache")
                return cached_response

        try:
            start_time = time.time()
            logger.debug(f"Calling OpenAI with model {self.model}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=build_default_messages(sanitized_prompt),
                response_format=response_format or None,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
            )

            result = response.choices[0].message.content
            elapsed_time = time.time() - start_time
            logger.debug(f"OpenAI response received in {elapsed_time:.2f}s")

            # Cache if appropriate
            if self._should_use_cache(sanitized_prompt, response_format, temperature):
                cache_key = generate_cache_key(sanitized_prompt, response_format, temperature, self.model)
                response_cache.put(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"OpenAILLMController Error: {str(e)}")
            if "rate limit" in str(e).lower():
                raise ModelOverloadedError(f"OpenAI rate limit exceeded: {str(e)}")
            elif "invalid" in str(e).lower() and "api key" in str(e).lower():
                raise ConnectionError(f"Invalid OpenAI API key: {str(e)}")
            elif "not found" in str(e).lower() and "model" in str(e).lower():
                raise ModelNotAvailableError(f"Model '{self.model}' not found: {str(e)}")
            else:
                raise LLMControllerError(f"OpenAI API error: {str(e)}")

    def get_completion_stream(
        self, prompt: str, response_format: dict, temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """
        Generates a streaming response from the OpenAI API.
        Currently collects chunks and returns the complete response.

        Args:
            prompt (str): The input query.
            response_format (dict): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The complete response
        """
        sanitized_prompt = sanitize_prompt(prompt)

        try:
            start_time = time.time()
            logger.debug("Calling OpenAI with streaming mode")

            # OpenAI supports true streaming, but we'll collect and return the full response
            # to maintain compatibility with the current interface
            response_chunks = []

            response = self.client.chat.completions.create(
                model=self.model,
                messages=build_default_messages(sanitized_prompt),
                response_format=response_format or None,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                stream=True,
            )

            # Collect chunks
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    response_chunks.append(chunk.choices[0].delta.content)

            full_response = "".join(response_chunks)
            elapsed_time = time.time() - start_time
            logger.debug(f"OpenAI streaming response completed in {elapsed_time:.2f}s")

            return full_response

        except Exception as e:
            logger.error(f"OpenAI streaming error: {str(e)}")
            # Fall back to non-streaming mode
            return self.get_completion(prompt, response_format, temperature)

    async def get_completion_async(
        self, prompt: str, response_format: dict, temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """
        Asynchronous version of get_completion for OpenAI.

        Args:
            prompt (str): The input query.
            response_format (dict): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The LLM response
        """
        sanitized_prompt = sanitize_prompt(prompt)

        try:
            import asyncio

            from openai import AsyncOpenAI

            # Create async client if needed
            if not hasattr(self, "async_client"):
                self.async_client = AsyncOpenAI(api_key=self.client.api_key)

            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=build_default_messages(sanitized_prompt),
                response_format=response_format or None,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
            )

            return response.choices[0].message.content

        except ImportError:
            # Fall back to thread-based async if AsyncOpenAI not available
            return await super().get_completion_async(prompt, response_format, temperature)
        except Exception as e:
            logger.error(f"OpenAI async error: {str(e)}")
            raise LLMControllerError(f"OpenAI async API error: {str(e)}")

    async def get_completion_stream_async(
        self, prompt: str, response_format: dict, temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """
        Asynchronous streaming completion for OpenAI.

        Args:
            prompt (str): The input query.
            response_format (dict): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The complete response
        """
        sanitized_prompt = sanitize_prompt(prompt)

        try:
            import asyncio

            from openai import AsyncOpenAI

            # Create async client if needed
            if not hasattr(self, "async_client"):
                self.async_client = AsyncOpenAI(api_key=self.client.api_key)

            response_chunks = []

            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=build_default_messages(sanitized_prompt),
                response_format=response_format or None,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    response_chunks.append(chunk.choices[0].delta.content)

            return "".join(response_chunks)

        except ImportError:
            # Fall back to thread-based async if AsyncOpenAI not available
            return await super().get_completion_stream_async(prompt, response_format, temperature)
        except Exception as e:
            logger.error(f"OpenAI async streaming error: {str(e)}")
            raise LLMControllerError(f"OpenAI async streaming error: {str(e)}")


# =============================================================================
# Unified LLMController
# =============================================================================
class LLMController:
    """
    Unified interface for interacting with various LLM backends.
    Defaults to Ollama (local) with "deepseek-r1:1.5b".

    Now supports chat_mode=True, which uses ChatOllamaLLMController for plain text,
    while leaving the original OllamaLLMController for JSON-based usage.

    Enhanced with streaming support and async capabilities.
    """

    BACKENDS: Dict[str, Type[AbstractLLMController]] = {"ollama": OllamaLLMController, "openai": OpenAILLMController}

    def __init__(
        self,
        backend: Literal["ollama", "openai"] = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        chat_mode: bool = False,
    ) -> None:
        """
        Initializes the unified LLM Controller.

        Args:
            backend (Literal["ollama", "openai"]): Which backend to use.
            model (Optional[str]): Model name (defaults to each controller's default).
            api_key (Optional[str]): For OpenAI usage if needed.
            chat_mode (bool): If True, we use ChatOllamaLLMController for plain text (ollama).
                              If False, we use standard JSON-based OllamaLLMController.

        Raises:
            ValueError: If an unsupported backend is specified.
            Import errors or connection errors from the underlying controllers.
        """
        if backend not in self.BACKENDS:
            raise ValueError(f"Backend must be one of: {', '.join(self.BACKENDS.keys())}")

        default_model = model or self.BACKENDS[backend].DEFAULT_MODEL

        if backend == "openai":
            # No chat_mode concept for OpenAI in this code, so we ignore it
            self.llm: AbstractLLMController = self.BACKENDS[backend](default_model, api_key)
            logger.info(f"LLMController using OpenAI backend with model '{default_model}'")
        else:
            # We're using "ollama"
            if chat_mode:
                # Use ChatOllamaLLMController
                self.llm = ChatOllamaLLMController(default_model)
                logger.info(f"LLMController using Ollama backend (chat mode) with model '{default_model}'")
            else:
                # Use the original JSON-based OllamaLLMController
                self.llm = OllamaLLMController(default_model)
                logger.info(f"LLMController using Ollama backend (JSON mode) with model '{default_model}'")

        # Store settings for reference
        self.backend = backend
        self.model = default_model
        self.chat_mode = chat_mode

    def get_completion(
        self, prompt: str, response_format: Optional[dict] = None, temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """
        Generates a response from the configured LLM backend.

        Args:
            prompt (str): The input query.
            response_format (Optional[dict]): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The LLM response as a JSON-formatted or plain string.

        Raises:
            Various LLMControllerError subclasses based on underlying controller.
        """
        try:
            sanitized_prompt = sanitize_prompt(prompt)
            return self.llm.get_completion(sanitized_prompt, response_format or {}, temperature)
        except LLMControllerError as e:
            logger.error(f"LLMController error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLMController: {str(e)}")
            raise LLMControllerError(f"Unexpected error: {str(e)}")

    async def get_completion_async(
        self, prompt: str, response_format: Optional[dict] = None, temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """
        Asynchronous version of get_completion.

        Args:
            prompt (str): The input query.
            response_format (Optional[dict]): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The LLM response as a JSON-formatted or plain string.

        Raises:
            Various LLMControllerError subclasses based on underlying controller.
        """
        try:
            sanitized_prompt = sanitize_prompt(prompt)
            return await self.llm.get_completion_async(sanitized_prompt, response_format or {}, temperature)
        except LLMControllerError as e:
            logger.error(f"LLMController error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLMController: {str(e)}")
            raise LLMControllerError(f"Unexpected error: {str(e)}")

    def get_completion_stream(
        self, prompt: str, response_format: Optional[dict] = None, temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """
        Generates a streaming response from the configured LLM backend.

        Args:
            prompt (str): The input query.
            response_format (Optional[dict]): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The LLM response as a JSON-formatted or plain string.

        Raises:
            Various LLMControllerError subclasses based on underlying controller.
        """
        try:
            sanitized_prompt = sanitize_prompt(prompt)
            return self.llm.get_completion_stream(sanitized_prompt, response_format or {}, temperature)
        except LLMControllerError as e:
            logger.error(f"LLMController error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLMController: {str(e)}")
            raise LLMControllerError(f"Unexpected error: {str(e)}")

    async def get_completion_stream_async(
        self, prompt: str, response_format: Optional[dict] = None, temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """
        Asynchronous version of get_completion_stream.

        Args:
            prompt (str): The input query.
            response_format (Optional[dict]): JSON schema or empty dict if none is needed.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The LLM response as a JSON-formatted or plain string.

        Raises:
            Various LLMControllerError subclasses based on underlying controller.
        """
        try:
            sanitized_prompt = sanitize_prompt(prompt)
            return await self.llm.get_completion_stream_async(sanitized_prompt, response_format or {}, temperature)
        except LLMControllerError as e:
            logger.error(f"LLMController error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLMController: {str(e)}")
            raise LLMControllerError(f"Unexpected error: {str(e)}")

    def check_health(self) -> Tuple[bool, str]:
        """
        Check the health status of the underlying LLM service.

        Returns:
            Tuple[bool, str]: A tuple of (is_healthy, status_message)
        """
        return self.llm.check_health()

    def clear_cache(self) -> None:
        """
        Clear the response cache for this controller.
        """
        response_cache.clear()
        logger.info("LLM response cache cleared")

    @staticmethod
    def get_cache_stats() -> Dict[str, int]:
        """
        Get statistics about the current cache usage.

        Returns:
            Dict[str, int]: Dictionary containing cache statistics
        """
        return {
            "size": len(response_cache.cache),
            "max_size": response_cache.max_size,
            "ttl_seconds": response_cache.ttl_seconds,
        }

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about the current backend configuration.

        Returns:
            Dict[str, Any]: Information about the current backend
        """
        return {"backend": self.backend, "model": self.model, "chat_mode": self.chat_mode}
