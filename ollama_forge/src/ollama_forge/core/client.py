import asyncio  # Added to fix "asyncio is not defined" warning
import json
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, AsyncIterator, Dict, Generator, Iterator, List, Optional, Union

import httpx
import requests
from tqdm.auto import tqdm  # Explicit import from tqdm.auto

from ollama_forge.src.ollama_forge.config.config import (
    API_ENDPOINTS,
    DEBUG_MODE,
    DEFAULT_OLLAMA_API_URL,
    DEFAULT_TIMEOUT,
    DISABLE_PROGRESS_BARS,
)
from ollama_forge.src.ollama_forge.config.model_constants import (
    get_fallback_model,
    resolve_model_alias,
)

# Internal imports
from ollama_forge.src.ollama_forge.core.exceptions import (
    ConnectionError,
    ModelNotFoundError,
    OllamaAPIError,
    ServerError,
    StreamingError,
    TimeoutError,
    get_exception_for_status,
)

# Utility flags and functions
TQDM_AVAILABLE = True
HELPERS_AVAILABLE = True  # Set to True if helper functions are available

# Set up module logger
logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with the Ollama API.

    This class provides a comprehensive interface to all Ollama API endpoints,
    with support for both synchronous and asynchronous requests, streaming responses,
    and robust error handling.

    Attributes:
        base_url: Base URL for the Ollama API
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for failed requests
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_API_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ):
        """
        Initialize the Ollama client.

        Args:
            base_url: Base URL for the Ollama API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            session: Optional requests.Session to use
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self._thread_local = threading.local()

    def _with_retry(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[requests.Response]:
        """
        Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint to call
            data: Request data
            stream: Whether to stream the response
            headers: Optional request headers

        Returns:
            Response object

        Raises:
            ConnectionError: If connection fails after all retries
            ModelNotFoundError: If the model is not found
            ServerError: If the server returns a 5xx error
            InvalidRequestError: If the request is invalid
            TimeoutError: If the request times out
            OllamaAPIError: For other API errors
        """
        url = f"{self.base_url}{endpoint}"
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        # Exponential backoff parameters
        backoff_factor = 0.5

        for attempt in range(self.max_retries + 1):
            try:
                if DEBUG_MODE and attempt > 0:
                    logger.debug(f"Retry attempt {attempt} for request to {url}")

                # Type-annotate response and ignore "json" param warning
                response: requests.Response = self.session.request(
                    method=method,
                    url=url,
                    json=data,  # type: ignore [call-arg]
                    headers=request_headers,
                    timeout=self.timeout,
                    stream=stream,
                )

                status_code: int = response.status_code  # Force known int type
                if status_code >= 400:
                    text: str = response.text  # Force known string type
                    error_message = f"HTTP {status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_message = error_data["error"]
                    except (ValueError, KeyError):
                        # If JSON parsing fails, use the response text
                        error_message = text or error_message

                    # Raise specific exception based on status code
                    exception = get_exception_for_status(
                        status_code,
                        error_message,
                        (
                            response.json()
                            if "json" in response.headers.get("content-type", "")
                            else None
                        ),
                    )
                    raise exception

                return response

            except requests.exceptions.Timeout:
                if attempt == self.max_retries:
                    raise TimeoutError(
                        f"Request to {url} timed out after {self.timeout}s and {self.max_retries} retries"
                    )

            except requests.exceptions.ConnectionError as e:
                if attempt == self.max_retries:
                    raise ConnectionError(
                        f"Connection to Ollama server failed: {str(e)}"
                    )

            except Exception as e:
                if attempt == self.max_retries:
                    raise OllamaAPIError(f"Unexpected error: {str(e)}")

                # Exponential backoff before retry
                wait_time = backoff_factor * (2**attempt)
                time.sleep(wait_time)

            # If we get here, all retries failed but no exception was raised
            return None  # Return None if we couldn't get a valid response

    async def _with_async_retry(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[httpx.Response]:
        url = f"{self.base_url}{endpoint}"
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        backoff_factor = 0.5

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    # Use explicit typing for the HTTP response
                    response: httpx.Response

                    # Make request based on method
                    if method == "POST":
                        response = await client.post(
                            url,
                            json=data,  # type: ignore [call-arg]
                            headers=request_headers,
                            timeout=self.timeout,
                            follow_redirects=True,
                        )
                    elif method == "GET":
                        response = await client.get(
                            url,
                            params=data,
                            headers=request_headers,
                            timeout=self.timeout,
                            follow_redirects=True,
                        )
                    elif method == "DELETE":
                        response = await client.delete(
                            url,
                            json=data,  # type: ignore [call-arg]
                            headers=request_headers,
                            timeout=self.timeout,
                            follow_redirects=True,
                        )
                    else:
                        raise OllamaAPIError(f"Unsupported method: {method}")

                    # Use explicit casting for status code
                    status_code = int(response.status_code)

                    if status_code >= 400:
                        # Explicitly type the error text
                        error_text = str(response.text)

                        if status_code == 404:
                            raise ModelNotFoundError(f"Model not found at {url}")
                        elif 400 <= status_code < 500:
                            raise OllamaAPIError(
                                f"Client error {status_code}: {error_text}"
                            )
                        else:
                            raise ServerError(
                                f"Server error {status_code}: {error_text}"
                            )

                    return response

            except (httpx.TimeoutException, httpx.RequestError) as e:
                if attempt == self.max_retries:
                    raise ConnectionError(f"Connection failed after retries: {e}")
                await asyncio.sleep(backoff_factor * (2**attempt))

        return None

    def get_version(self) -> Dict[str, Any]:
        """
        Get the Ollama server version.

        Returns:
            Dictionary with version information

        Raises:
            ConnectionError: If cannot connect to Ollama server
            OllamaAPIError: If the response is invalid
        """
        endpoint = API_ENDPOINTS["version"]
        response = self._with_retry("GET", endpoint)
        if response is None:
            raise OllamaAPIError("Failed to get Ollama version: No response received")
        return response.json()

    def list_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List available models.

        Returns:
            Dictionary with models information

        Raises:
            ConnectionError: If cannot connect to Ollama server
            OllamaAPIError: If the response is invalid
        """
        endpoint = API_ENDPOINTS["tags"]
        response = self._with_retry("GET", endpoint)
        if response is None:
            raise OllamaAPIError("Failed to list models: No response received")
        return response.json()

    def pull_model(
        self, model: str, stream: bool = True
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Pull a model from the Ollama registry.

        Args:
            model: Name of the model to pull
            stream: Whether to stream the progress

        Returns:
            If stream=True, a generator yielding progress updates
            If stream=False, a dictionary with pull result

        Raises:
            ConnectionError: If cannot connect to Ollama server
            InvalidRequestError: If the model name is invalid
            OllamaAPIError: If the response is invalid
        """
        endpoint = API_ENDPOINTS["pull"]
        data: Dict[str, Any] = {"name": model}

        if not stream:
            response = self._with_retry("POST", endpoint, data=data)
            if response is None:
                raise OllamaAPIError(f"Failed to pull model '{model}'")
            return response.json()

        # Stream the progress
        response = self._with_retry("POST", endpoint, data=data, stream=True)
        if response is None:
            raise OllamaAPIError(f"Failed to pull model '{model}' with streaming")

        def generate_updates() -> Generator[Dict[str, Any], None, None]:
            progress_bar = None
            last_status = None

            try:
                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        progress = json.loads(line)

                        # Update progress bar if available
                        if TQDM_AVAILABLE and not DISABLE_PROGRESS_BARS:
                            if "total" in progress and "completed" in progress:
                                if progress_bar is None:
                                    progress_bar = tqdm(
                                        total=progress["total"],
                                        desc=f"Pulling {model}",
                                        unit="B",
                                        unit_scale=True,
                                    )

                                # Update progress
                                if progress["completed"] > progress_bar.n:
                                    progress_bar.update(
                                        progress["completed"] - progress_bar.n
                                    )

                        # Only yield status changes to avoid flooding logs
                        if "status" in progress:
                            if progress["status"] != last_status:
                                last_status = progress["status"]
                                yield progress
                        else:
                            yield progress

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse progress line: {line}")

            finally:
                # Clean up progress bar
                if progress_bar is not None:
                    progress_bar.close()

        return generate_updates()

    def generate(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate text from a prompt.

        Args:
            model: Name of the model to use
            prompt: The prompt to generate from
            options: Dictionary of generation options
            stream: Whether to stream the response

        Returns:
            If stream=True, a generator yielding response chunks
            If stream=False, a dictionary with the complete response

        Raises:
            ConnectionError: If cannot connect to Ollama server
            ModelNotFoundError: If the model does not exist
            InvalidRequestError: If the request is invalid
        """
        endpoint = API_ENDPOINTS["generate"]
        data: Dict[str, Any] = {
            "model": resolve_model_alias(model) if HELPERS_AVAILABLE else model,
            "prompt": prompt,
            "stream": stream,
        }

        # Add optional parameters
        if options:
            for key, value in options.items():
                data[key] = value

        if not stream:
            # Single response
            response = self._with_retry("POST", endpoint, data=data)
            if response is None:
                raise OllamaAPIError(f"Failed to generate text with model '{model}'")
            return response.json()

        # Stream responses
        response = self._with_retry("POST", endpoint, data=data, stream=True)
        if response is None:
            raise OllamaAPIError(
                f"Failed to generate streaming text with model '{model}'"
            )

        def generate_chunks() -> Iterator[Dict[str, Any]]:
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                    yield chunk
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse response chunk: {line}")
                    raise StreamingError("Failed to parse streaming response")

        return generate_chunks()

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Chat with a model.

        Args:
            model: Name of the model
            messages: List of message dictionaries (role, content)
            options: Chat options
            stream: Whether to stream the response

        Returns:
            If stream=True, a generator yielding response chunks
            If stream=False, a dictionary with the complete response

        Raises:
            ConnectionError: If cannot connect to Ollama server
            ModelNotFoundError: If the model does not exist
            InvalidRequestError: If the messages format is invalid
        """
        endpoint = API_ENDPOINTS["chat"]
        data: Dict[str, Any] = {
            "model": resolve_model_alias(model) if HELPERS_AVAILABLE else model,
            "messages": messages,
            "stream": stream,
        }

        # Add optional parameters
        if options:
            for key, value in options.items():
                data[key] = value

        if not stream:
            # Single response
            response = self._with_retry("POST", endpoint, data=data)
            if response is None:
                raise OllamaAPIError(f"Failed to chat with model '{model}'")
            return response.json()

        # Stream responses
        response = self._with_retry("POST", endpoint, data=data, stream=True)
        if response is None:
            raise OllamaAPIError(f"Failed to stream chat with model '{model}'")

        def generate_chunks() -> Iterator[Dict[str, Any]]:
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                    yield chunk
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse chat response chunk: {line}")
                    raise StreamingError("Failed to parse streaming chat response")

        return generate_chunks()

    def create_embedding(
        self, model: str, prompt: str, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an embedding vector for a text prompt.

        Args:
            model: Name of the model
            prompt: Text to create embedding for
            options: Optional embedding parameters

        Returns:
            Dictionary with the embedding vector

        Raises:
            ConnectionError: If cannot connect to Ollama server
            ModelNotFoundError: If the model does not exist
        """
        endpoint = API_ENDPOINTS["embedding"]
        data: Dict[str, Any] = {
            "model": resolve_model_alias(model) if HELPERS_AVAILABLE else model,
            "prompt": prompt,
        }

        # Add optional parameters
        if options:
            for key, value in options.items():
                data[key] = value

        response = self._with_retry("POST", endpoint, data=data)
        if response is None:
            raise OllamaAPIError(f"Failed to create embedding with model '{model}'")

        # Get the response data
        response_data = response.json()

        # Add the 'embedding' key if it doesn't exist but 'embeddings' does
        if "embeddings" in response_data and "embedding" not in response_data:
            if response_data["embeddings"] and len(response_data["embeddings"]) > 0:
                response_data["embedding"] = response_data["embeddings"][0]

        return response_data

    def batch_embeddings(
        self,
        model: str,
        prompts: List[str],
        options: Optional[Dict[str, Any]] = None,
        show_progress: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Create embeddings for multiple prompts.

        Args:
            model: Name of the model
            prompts: List of texts to create embeddings for
            options: Optional embedding parameters
            show_progress: Whether to show a progress bar

        Returns:
            List of embedding vectors

        Raises:
            ConnectionError: If cannot connect to Ollama server
            ModelNotFoundError: If the model does not exist
        """
        results: List[Dict[str, Any]] = []

        # Prepare progress bar
        if show_progress and TQDM_AVAILABLE and not DISABLE_PROGRESS_BARS:
            iterator = tqdm(prompts, desc=f"Creating embeddings with {model}")
        else:
            iterator = prompts

        for prompt in iterator:
            embedding = self.create_embedding(model, prompt, options)
            results.append(embedding)

        return results

    def delete_model(self, model: str) -> bool:
        """
        Delete a model.

        Args:
            model: Name of the model to delete

        Returns:
            True if successful

        Raises:
            ConnectionError: If cannot connect to Ollama server
            ModelNotFoundError: If the model does not exist
        """
        endpoint = API_ENDPOINTS["delete"]
        data: Dict[str, Any] = {"model": model}

        response = self._with_retry("DELETE", endpoint, data=data)
        if response is None:
            raise OllamaAPIError(f"Failed to delete model '{model}'")
        return response.status_code == 200

    def copy_model(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Copy a model.

        Args:
            source: Source model name
            destination: Destination model name

        Returns:
            Dictionary with operation result

        Raises:
            ConnectionError: If cannot connect to Ollama server
            ModelNotFoundError: If the source model does not exist
            InvalidRequestError: If the destination name is invalid
        """
        endpoint = API_ENDPOINTS["copy"]
        data: Dict[str, Any] = {"source": source, "destination": destination}

        response = self._with_retry("POST", endpoint, data=data)
        if response is None:
            raise OllamaAPIError(
                f"Failed to copy model from '{source}' to '{destination}'"
            )
        return response.json()

    def create_model(
        self, name: str, modelfile: str, stream: bool = True
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Create a new model from a Modelfile.

        Args:
            name: Name for the new model
            modelfile: Modelfile content
            stream: Whether to stream the creation progress

        Returns:
            If stream=True, a generator yielding progress updates
            If stream=False, a dictionary with creation result

        Raises:
            ConnectionError: If cannot connect to Ollama server
            InvalidRequestError: If the Modelfile is invalid
        """
        endpoint = API_ENDPOINTS["create"]
        data: Dict[str, Any] = {"name": name, "modelfile": modelfile}

        if not stream:
            response = self._with_retry("POST", endpoint, data=data)
            if response is None:
                raise OllamaAPIError(f"Failed to create model '{name}'")
            return response.json()

        # Stream the progress
        response = self._with_retry("POST", endpoint, data=data, stream=True)
        if response is None:
            raise OllamaAPIError(f"Failed to create model '{name}' with streaming")

        def generate_updates() -> Iterator[Dict[str, Any]]:
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    update = json.loads(line)
                    yield update
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse creation progress: {line}")
                    raise StreamingError("Failed to parse streaming response")

        return generate_updates()

    @contextmanager
    def fallback_context(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Context manager for automatic model fallback.

        Args:
            prompt: The user's query or message
            system_prompt: Optional system prompt to use

        Returns:
            Context dictionary with parameters

        Example:
            ```python
            with client.fallback_context("chat"):
                response = client.chat(model, messages)
            ```
        """
        if not hasattr(self._thread_local, "fallback_depth"):
            self._thread_local.fallback_depth = 0

        self._thread_local.fallback_depth += 1

        try:
            yield
        except (ModelNotFoundError, ServerError) as e:
            # Only attempt fallback if not already in a fallback operation
            if self._thread_local.fallback_depth <= 1 and HELPERS_AVAILABLE:
                model = (
                    e.response.get("model")
                    if hasattr(e, "response") and e.response
                    else None
                )

                if model:
                    fallback_model = get_fallback_model(model, operation)

                    # Log the fallback
                    logger.warning(f"Falling back from {model} to {fallback_model}")

                    # Store the original exception for reference
                    self._thread_local.original_error = e

                    # The calling code should handle fallback by checking thread_local
                    self._thread_local.fallback_model = fallback_model
                    return

            # Re-raise the exception if we can't handle the fallback
            raise
        finally:
            self._thread_local.fallback_depth -= 1

    def get_fallback_info(self) -> Dict[str, Any]:
        """
        Get information about the current fallback state.

        Returns:
            Dictionary containing fallback depth, model name, and original error
        """
        return {
            "depth": getattr(self._thread_local, "fallback_depth", 0),
            "model": getattr(self._thread_local, "fallback_model", None),
            "original_error": getattr(self._thread_local, "original_error", None),
        }

    # Async convenience methods for easy transition to AsyncOllamaClient
    async def agenerate(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        data: Dict[str, Any] = {
            "model": resolve_model_alias(model) if HELPERS_AVAILABLE else model,
            "prompt": prompt,
            "stream": stream,
        }
        if options:
            data.update(options)

        if not stream:
            response = await self._with_async_retry(
                "POST", API_ENDPOINTS["generate"], data=data
            )
            if response is None:
                raise OllamaAPIError(f"agenerate failed for model '{model}'")
            return response.json()

        response = await self._with_async_retry(
            "POST", API_ENDPOINTS["generate"], data=data, stream=True
        )
        if response is None:
            raise OllamaAPIError(f"Streaming agenerate failed for model '{model}'")

        async def generate_chunks() -> AsyncIterator[Dict[str, Any]]:
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    raise StreamingError(f"Failed to parse streamed line: {line}")

        return generate_chunks()

    async def achat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        data: Dict[str, Any] = {
            "model": resolve_model_alias(model) if HELPERS_AVAILABLE else model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            data.update(options)

        if not stream:
            response = await self._with_async_retry(
                "POST", API_ENDPOINTS["chat"], data=data
            )
            if response is None:
                raise OllamaAPIError(f"achat failed for model '{model}'")
            return response.json()

        response = await self._with_async_retry(
            "POST", API_ENDPOINTS["chat"], data=data, stream=True
        )
        if response is None:
            raise OllamaAPIError(f"Streaming achat failed for model '{model}'")

        async def generate_chunks() -> AsyncIterator[Dict[str, Any]]:
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    raise StreamingError(f"Failed to parse streamed line: {line}")

        return generate_chunks()

    async def acreate_embedding(
        self, model: str, prompt: str, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model": resolve_model_alias(model) if HELPERS_AVAILABLE else model,
            "prompt": prompt,
        }
        if options:
            data.update(options)

        response = await self._with_async_retry(
            "POST", API_ENDPOINTS["embedding"], data=data
        )
        if response is None:
            raise OllamaAPIError(f"acreate_embedding failed for model '{model}'")
        return response.json()
