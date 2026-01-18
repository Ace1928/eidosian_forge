#!/usr/bin/env python3
"""
Common utilities for Ollama Forge examples and client.

This module provides a collection of helper functions for interacting with the Ollama API,
checking installation status, and managing the Ollama service lifecycle.
"""
import json
import logging
import os
import platform
import subprocess
import time
from typing import Any, Dict, Optional, Tuple, TypeVar, Union

import aiohttp
import requests
from colorama import Fore, Style

# Configure logger with module-level scope
logger = logging.getLogger(__name__)

# Core configuration constants
DEFAULT_OLLAMA_API_URL = "http://localhost:11434/"
DEFAULT_MODEL = "deepseek-r1:1.5b"
ALTERNATIVE_MODEL = "qwen2.5:0.5b"

# Type variable for generic return types
T = TypeVar("T")
ResponseType = Union[Dict[str, Any], requests.Response, aiohttp.ClientResponse]


def print_header(title: str) -> None:
    """Print a formatted header with the given title.

    Args:
        title: The text to display in the header
    """
    print(f"\n{Fore.CYAN}=== {title} ==={Style.RESET_ALL}\n")


def print_success(message: str) -> None:
    """Print a success message with a green checkmark.

    Args:
        message: The success message to display
    """
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def print_error(message: str) -> None:
    """Print an error message with a red X.

    Args:
        message: The error message to display
    """
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")


def print_warning(message: str) -> None:
    """Print a warning message with a yellow warning symbol.

    Args:
        message: The warning message to display
    """
    print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")


def print_info(message: str) -> None:
    """Print an informational message with a blue info symbol.

    Args:
        message: The informational message to display
    """
    print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")


def print_json(data: Any) -> None:
    """Print JSON data in a formatted, readable way.

    Args:
        data: The data structure to format and print as JSON
    """
    print(json.dumps(data, indent=2, default=str))


def make_api_request(
    method: str,
    endpoint: str,
    data: Optional[Dict[str, Any]] = None,
    base_url: str = DEFAULT_OLLAMA_API_URL,
    timeout: int = 60,
) -> ResponseType:
    """Make a request to the Ollama API with elegant error handling.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint to call (with or without leading slash)
        data: JSON data to send (for POST, PUT, etc.)
        base_url: Base URL for the API
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response, raw response object, or error details dictionary
    """
    url = f"{base_url.rstrip('/')}{endpoint}"
    session = requests.Session()

    try:
        headers = {"Content-Type": "application/json"}
        response = session.request(
            method=method,
            url=url,
            json=data if data else None,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return response
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")

        # Transform into a structured error response
        error_data = {
            "error": type(e).__name__,
            "message": str(e),
            "url": url,
            "method": method,
        }

        # Enhance error details for specific error types
        if isinstance(e, requests.exceptions.Timeout):
            error_data["timeout"] = str(timeout)
            error_data["suggestion"] = "Consider increasing the timeout value"

        logger.debug(f"Structured error: {error_data}")
        return error_data


async def async_make_api_request(
    method: str,
    endpoint: str,
    data: Optional[Dict[str, Any]] = None,
    base_url: str = DEFAULT_OLLAMA_API_URL,
    timeout: int = 60,
) -> ResponseType:
    """Make an asynchronous request to the Ollama API.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint to call (with or without leading slash)
        data: JSON data to send
        base_url: Base URL for the API
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response, raw response object, or error details dictionary
    """
    url = f"{base_url.rstrip('/')}{endpoint}"
    timeout_obj = aiohttp.ClientTimeout(total=timeout)

    try:
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.request(
                method=method,
                url=url,
                json=data if data else None,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    return {
                        "error": f"HTTP {response.status}",
                        "message": error_text,
                        "url": url,
                        "method": method,
                    }

                try:
                    return await response.json()
                except ValueError:
                    return response
    except aiohttp.ClientError as e:
        logger.error(f"Async API request failed: {e}")
        return {
            "error": type(e).__name__,
            "message": str(e),
            "url": url,
            "method": method,
        }


def check_ollama_installed() -> bool:
    """Check if Ollama is installed on the system.

    Checks appropriate locations based on platform (Windows, macOS, Linux).

    Returns:
        True if installed, False otherwise
    """
    system = platform.system().lower()

    try:
        if system == "windows":
            # Check for Ollama in standard Windows installation locations
            paths = [
                os.path.expandvars(r"%ProgramFiles%\Ollama"),
                os.path.expandvars(r"%LOCALAPPDATA%\Ollama"),
            ]
            return any(os.path.exists(path) for path in paths)
        else:
            # For Unix-based systems, check binary in PATH
            result = subprocess.run(
                ["which", "ollama"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return result.returncode == 0
    except Exception as e:
        logger.warning(f"Error checking Ollama installation: {e}")
        return False


def check_ollama_running() -> Tuple[bool, str]:
    """Check if the Ollama server is running.

    Makes a request to the Ollama API version endpoint with a short timeout.

    Returns:
        Tuple of (is_running, message) where message provides context
    """
    try:
        response = requests.get(
            f"{DEFAULT_OLLAMA_API_URL.rstrip('/')}/api/version", timeout=2.0
        )
        if response.status_code == 200:
            version_info = response.json()
            version = version_info.get("version", "unknown")
            return True, f"Ollama server is running (version {version})"
        return False, f"Ollama server returned status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to Ollama server"
    except requests.exceptions.Timeout:
        return False, "Connection to Ollama server timed out"
    except Exception as e:
        return False, f"Error checking Ollama server: {str(e)}"


def get_ollama_version() -> str:
    """Get the version of the running Ollama server.

    Makes a request to the Ollama API version endpoint.

    Returns:
        Version string or "unknown" if version cannot be determined
    """
    try:
        response = requests.get(
            f"{DEFAULT_OLLAMA_API_URL.rstrip('/')}/api/version", timeout=2.0
        )
        if response.status_code == 200:
            version_info = response.json()
            return version_info.get("version", "unknown")
        return "unknown"
    except Exception:
        return "unknown"


def install_ollama(target_dir: Optional[str] = None) -> Tuple[bool, str]:
    """Install Ollama on the system if not already installed.

    Uses platform-appropriate methods: official install script for Unix systems,
    and instructions for Windows.

    Args:
        target_dir: Optional directory to install Ollama into (currently unused)

    Returns:
        Tuple of (success, message)
    """
    if target_dir:
        logger.warning(
            f"Target directory '{target_dir}' specified but not currently used"
        )

    system = platform.system().lower()

    if check_ollama_installed():
        return True, "Ollama is already installed"

    try:
        if system in ("linux", "darwin"):
            # Use the official install script for Linux/macOS
            subprocess.run(
                ["curl", "-fsSL", "https://ollama.com/install.sh", "|", "sh"],
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return True, f"Successfully installed Ollama on {system}"
        elif system == "windows":
            # Windows installation requires manual steps
            return (
                False,
                "Windows installation must be done manually from https://ollama.com/download",
            )
        else:
            return False, f"Unsupported platform: {system}"
    except subprocess.CalledProcessError as e:
        return False, f"Installation failed: {e.stderr}"


def ensure_ollama_running() -> Tuple[bool, str]:
    """Ensure Ollama server is running, starting it if needed.

    First checks if Ollama is running, then attempts to start it if not.
    On Unix systems, this launches the server as a background process.
    On Windows, this provides instructions for manual startup.

    Returns:
        Tuple of (success, message)
    """
    # First check if Ollama is already running
    is_running, _ = check_ollama_running()
    if is_running:
        version = get_ollama_version()
        return True, f"Ollama is ready (version {version})"

    # Not running, try to start it
    system = platform.system().lower()
    try:
        if system in ("linux", "darwin"):
            # Start the Ollama service
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            # Wait for it to be ready with exponential backoff
            for attempt in range(5):
                is_running, _ = check_ollama_running()
                if is_running:
                    return True, "Ollama started successfully"
                # Progressively longer waits (0.5s, 1s, 2s, 4s, 8s)
                time.sleep(0.5 * (2**attempt))

            return False, "Started Ollama but service did not become ready"
        elif system == "windows":
            return False, "Starting Ollama on Windows must be done manually"
        else:
            return False, f"Unsupported platform for starting Ollama: {system}"
    except Exception as e:
        return False, f"Error starting Ollama: {str(e)}"
