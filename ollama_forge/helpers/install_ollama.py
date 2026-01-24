"""Ollama installation helper utilities."""

import subprocess
import sys
from typing import Optional


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


def check_ollama_running(url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running."""
    import httpx
    try:
        with httpx.Client() as client:
            resp = client.get(f"{url}/api/tags", timeout=5.0)
            return resp.status_code == 200
    except Exception:
        return False


def install_ollama() -> bool:
    """Install Ollama (Linux/Mac only)."""
    try:
        result = subprocess.run(
            ["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"],
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode == 0
    except Exception:
        return False


def pull_model(model_name: str) -> bool:
    """Pull an Ollama model."""
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=600
        )
        return result.returncode == 0
    except Exception:
        return False


def ensure_ollama_running(url: str = "http://localhost:11434") -> bool:
    """Ensure Ollama is running, attempt to start if not."""
    if check_ollama_running(url):
        return True
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
