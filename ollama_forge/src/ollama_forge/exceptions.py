"""Custom exceptions for ollama_forge."""


class OllamaError(Exception):
    """Base exception for Ollama-related errors."""
    pass


class OllamaAPIError(OllamaError):
    """Error from Ollama API call."""
    def __init__(self, message: str, status_code: int = None, response: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class ModelNotFoundError(OllamaError):
    """Model not found in Ollama."""
    def __init__(self, model_name: str, available_models: list = None):
        msg = f"Model '{model_name}' not found"
        if available_models:
            msg += f". Available models: {', '.join(available_models)}"
        super().__init__(msg)
        self.model_name = model_name
        self.available_models = available_models or []


class ConnectionError(OllamaError):
    """Cannot connect to Ollama server."""
    def __init__(self, url: str, original_error: Exception = None):
        msg = f"Cannot connect to Ollama at {url}"
        if original_error:
            msg += f": {original_error}"
        super().__init__(msg)
        self.url = url
        self.original_error = original_error
