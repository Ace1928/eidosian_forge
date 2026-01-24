"""Common utilities for ollama_forge."""

import httpx
import json
import subprocess
from typing import Any, Dict, Optional

DEFAULT_OLLAMA_API_URL = "http://localhost:11434"


def print_header(text: str) -> None:
    """Print a styled header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_success(text: str) -> None:
    """Print success message in green."""
    print(f"✅ {text}")


def print_error(text: str) -> None:
    """Print error message in red."""
    print(f"❌ {text}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"ℹ️  {text}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"⚠️  {text}")


def print_json(data: Any, indent: int = 2) -> None:
    """Print formatted JSON."""
    print(json.dumps(data, indent=indent, default=str))


def make_api_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict] = None,
    base_url: str = DEFAULT_OLLAMA_API_URL,
    timeout: float = 30.0,
) -> Optional[Dict]:
    """Make an API request to Ollama."""
    url = f"{base_url}{endpoint}"
    try:
        with httpx.Client() as client:
            if method.upper() == "GET":
                resp = client.get(url, timeout=timeout)
            else:
                resp = client.post(url, json=data, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print_error(f"API request failed: {e}")
        return None


def check_ollama_running(url: str = DEFAULT_OLLAMA_API_URL) -> bool:
    """Check if Ollama server is running."""
    try:
        with httpx.Client() as client:
            resp = client.get(f"{url}/api/tags", timeout=5.0)
            return resp.status_code == 200
    except Exception:
        return False


def check_ollama_installed() -> bool:
    """Check if Ollama is installed and accessible."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def ensure_ollama_running(url: str = DEFAULT_OLLAMA_API_URL) -> bool:
    """Ensure Ollama is running, attempt to start if not."""
    if check_ollama_running(url):
        return True
    print_info("Ollama not running, attempting to start...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import time
        time.sleep(2)
        return check_ollama_running(url)
    except Exception:
        return False
