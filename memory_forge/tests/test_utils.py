import json
import logging
import time
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

from llm_controller import (
    LLMControllerError,
    ModelNotAvailableError,
    InvalidParameterError,
    ConnectionError,
    ResponseParseError,
    ModelOverloadedError,
    AbstractLLMController
)

logger = logging.getLogger(__name__)

class MockLLMController(AbstractLLMController):
    """
    A mock LLM controller for testing purposes.
    
    This controller simulates the behavior of a real LLM controller by returning
    predefined responses, simulating errors, and tracking calls.
    """
    
    def __init__(self) -> None:
        """Initialize the mock controller with default settings."""
        # Default mock response - a JSON string for evolution responses
        default_response = {
            "should_evolve": False,
            "evolution_type": [],
            "reasoning": "This is a mock response",
            "affected_units": [],
            "evolution_details": {
                "new_context": "",
                "new_keywords": [],
                "new_relationships": []
            }
        }
        
        self.mock_response = json.dumps(default_response)
        self.calls = []
        self._response_sequence = []
        self._response_index = 0
        self._errors = []
        self._error_index = 0
        self._healthy = True
        self._health_message = "Mock LLM is healthy"
        self._cache = {}

    def get_completion(
        self, 
        prompt: str, 
        response_format: dict = None, 
        temperature: float = 0.7
    ) -> str:
        """
        Simulate an LLM completion call.
        
        Args:
            prompt: The input prompt
            response_format: Expected response format
            temperature: Sampling temperature
            
        Returns:
            A string response (usually JSON formatted)
            
        Raises:
            Various exceptions if error simulation is enabled
        """
        # Track the call
        self.calls.append({
            "prompt": prompt,
            "response_format": response_format,
            "temperature": temperature,
            "timestamp": time.time()
        })
        
        # Check if we should throw an error
        if self._errors and self._error_index < len(self._errors):
            error = self._errors[self._error_index]
            self._error_index += 1
            raise error
        
        # Check cache for deterministic requests
        if temperature < 0.1:
            cache_key = f"{prompt}_{response_format}_{temperature}"
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Determine which response to return
        if self._response_sequence and self._response_index < len(self._response_sequence):
            response = self._response_sequence[self._response_index]
            self._response_index += 1
        else:
            response = self.mock_response
            
        # Cache deterministic responses
        if temperature < 0.1:
            cache_key = f"{prompt}_{response_format}_{temperature}"
            self._cache[cache_key] = response
            
        return response

    def get_completion_stream(
        self, 
        prompt: str, 
        response_format: dict = None, 
        temperature: float = 0.7
    ) -> str:
        """
        Simulate streaming completion.
        
        Args:
            prompt: Input prompt
            response_format: Response format specification
            temperature: Sampling temperature
            
        Returns:
            Complete response string
        """
        # Just use regular completion for simplicity
        return self.get_completion(prompt, response_format, temperature)
    
    def simulate_streaming(self, prompt: str, callback: Callable[[str, bool], None]) -> str:
        """
        Simulate streaming by calling the callback with chunks.
        
        Args:
            prompt: Input prompt
            callback: Function to call with each chunk and completion flag
            
        Returns:
            Full response
        """
        # Get the full response
        full_response = self.get_completion(prompt, {})
        
        # Break it into chunks
        chunk_size = 5
        for i in range(0, len(full_response), chunk_size):
            chunk = full_response[i:i+chunk_size]
            is_done = i + chunk_size >= len(full_response)
            callback(chunk, is_done)
            time.sleep(0.01)  # Small delay between chunks
            
        return full_response

    def check_health(self) -> Tuple[bool, str]:
        """
        Mock health check.
        
        Returns:
            Tuple of (is_healthy, message)
        """
        return self._healthy, self._health_message
    
    def set_health_status(self, is_healthy: bool, message: str = "") -> None:
        """
        Set the mock health status.
        
        Args:
            is_healthy: Whether the mock is "healthy"
            message: Status message
        """
        self._healthy = is_healthy
        self._health_message = message or ("Mock LLM is healthy" if is_healthy else "Mock LLM is unhealthy")
    
    def set_responses(self, responses: List[str]) -> None:
        """
        Set a sequence of responses to return.
        
        Args:
            responses: List of response strings
        """
        self._response_sequence = responses
        self._response_index = 0
    
    def set_errors(self, errors: List[Exception]) -> None:
        """
        Set a sequence of errors to throw.
        
        Args:
            errors: List of exceptions to raise
        """
        self._errors = errors
        self._error_index = 0
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache = {}
