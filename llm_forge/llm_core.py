"""
LLM Forge - Language model orchestration and inference optimization.
Centralized coordination point for all language model interactions.
"""
import requests
import logging
import time
import os
import subprocess
import socket
import shutil
from typing import Dict, Any, List, Optional

class LLMForge:
    """
    Orchestrates interactions with local LLM backends.
    Supports parameter tuning, structured output, and smart fallback with auto-launch capabilities.
    """
    def __init__(self, base_url: str = "http://localhost:11434", use_codex: bool = False):
        """
        Initialize LLMForge.
        
        Args:
            base_url: The primary URL to connect to.
            use_codex: If True, treats the primary URL as a ChatMock/Codex endpoint.
                       If False (default), assumes standard Ollama API.
        """
        self.use_codex = use_codex
        
        if self.use_codex:
            self.primary_url = base_url
            self.primary_model = "gpt-5" # Default for Codex/ChatMock
            # Fallback configuration
            self.fallback_url = os.getenv("OLLAMA_FALLBACK_URL", "http://localhost:11434")
            self.fallback_model = os.getenv("OLLAMA_FALLBACK_MODEL") 
        else:
            self.primary_url = base_url if base_url else "http://localhost:11434"
            self.primary_model = os.getenv("OLLAMA_MODEL") 
            self.fallback_url = None # No fallback needed if we are already local
            self.fallback_model = None

        # Preferred local models in order of priority (lightweight & performant)
        self.local_model_priority = [
            "qwen2.5:1.5b-Instruct", # New default preference
            "qwen2.5:0.5b-Instruct",
            "qwen2.5:1.5b",
            "qwen2.5:0.5b",
            "qwen:0.5b",
            "gemma3:1b",
            "deepseek-r1:1.5b",
            "tinyllama",
            "llama3.2:1b"
        ]
        
        # Session for connection pooling
        self.session = requests.Session()

    def _is_port_open(self, url: str) -> bool:
        """Check if the port for the given URL is open."""
        try:
            if ":" not in url.split("//")[-1]:
                port = 80 if url.startswith("http://") else 443
                host = url.split("//")[-1].split("/")[0]
            else:
                host_port = url.split("//")[-1].split("/")[0]
                host, port_str = host_port.split(":")
                port = int(port_str)
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5) # Fast timeout for checks
                return s.connect_ex((host, port)) == 0
        except Exception:
            return False

    def _launch_ollama(self) -> bool:
        """Attempt to launch Ollama if found in PATH."""
        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            logging.error("Ollama binary not found in PATH.")
            return False
            
        logging.info("Launching Ollama server...")
        try:
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            # Wait for it to come up
            target_url = self.fallback_url if self.use_codex else self.primary_url
            for _ in range(10):
                time.sleep(1)
                if self._is_port_open(target_url):
                    logging.info("Ollama started successfully.")
                    return True
            logging.error("Ollama failed to start within timeout.")
            return False
        except Exception as e:
            logging.error(f"Failed to launch Ollama: {e}")
            return False

    def _ensure_local_ready(self, url: str) -> bool:
        """Check if local Ollama is running, launch if not."""
        if self._is_port_open(url):
            return True
        return self._launch_ollama()

    def _get_available_models(self, url: str) -> List[str]:
        try:
            resp = self.session.get(f"{url}/api/tags", timeout=2)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    def _select_best_local_model(self, url: str, current_selection: Optional[str] = None) -> str:
        """Find the best available model from the priority list."""
        if current_selection:
            return current_selection
            
        available = self._get_available_models(url)
        
        # Check exact matches from priority list
        for candidate in self.local_model_priority:
            if candidate in available:
                logging.info(f"Selected local model: {candidate}")
                return candidate
        
        # Check partial matches
        for candidate in self.local_model_priority:
            base_name = candidate.split(":")[0]
            for avail in available:
                if base_name in avail:
                     logging.info(f"Selected partial match model: {avail} (preferred {candidate})")
                     return avail

        if available:
            logging.warning(f"No priority model found. Using first available: {available[0]}")
            return available[0]
            
        logging.error("No models found in Ollama. Please run 'ollama pull qwen2.5:1.5b-Instruct'")
        return "qwen2.5:1.5b-Instruct" # Default hope that it might exist or aut-pull

    def _call_generate(self, url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        """Helper to make the actual API call."""
        response = self.session.post(f"{url}/api/generate", json=payload, timeout=timeout)
        if response.status_code == 429:
            raise requests.exceptions.RequestException("Rate limit exceeded (429)")
        response.raise_for_status()
        return response.json()

    def generate(self, prompt: str, model: Optional[str] = None, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None, json_mode: bool = False) -> Dict[str, Any]:
        """
        Generate a response.
        If use_codex=True, tries ChatMock first, then falls back to local Ollama.
        If use_codex=False, uses local Ollama directly.
        """
        # Determine initial target
        if self.use_codex:
            target_url = self.primary_url
            target_model = model or self.primary_model
        else:
            target_url = self.primary_url
            # For local, ensure we pick a good model if none specified
            if not self._ensure_local_ready(target_url):
                 return {"success": False, "error": "Local Ollama server could not be started."}
            
            target_model = model or self.primary_model or self._select_best_local_model(target_url)
            self.primary_model = target_model # Cache selection

        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": options or {}
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        try:
            data = self._call_generate(target_url, payload)
            return {
                "response": data.get("response", ""),
                "success": True,
                "model": target_model,
                "metadata": {
                    "total_duration": data.get("total_duration"),
                    "eval_count": data.get("eval_count")
                }
            }
        except Exception as e:
            # Fallback Logic (Only if using Codex primarily)
            if self.use_codex:
                logging.warning(f"Primary provider ({target_url}) failed: {e}. Attempting local fallback...")
                
                if not self._ensure_local_ready(self.fallback_url):
                    return {"success": False, "error": f"Primary failed ({e}) and local fallback unavailable."}
                
                fallback_model = self._select_best_local_model(self.fallback_url, self.fallback_model)
                self.fallback_model = fallback_model # Cache selection
                
                logging.info(f"Falling back to {fallback_model} at {self.fallback_url}")
                
                fallback_payload = payload.copy()
                fallback_payload["model"] = fallback_model
                
                try:
                    data = self._call_generate(self.fallback_url, fallback_payload)
                    return {
                        "response": data.get("response", ""),
                        "success": True,
                        "model": fallback_model,
                        "fallback_used": True,
                        "metadata": {
                            "total_duration": data.get("total_duration"),
                            "eval_count": data.get("eval_count")
                        }
                    }
                except Exception as fb_e:
                    logging.error(f"Fallback generation failed: {fb_e}")
                    return {"success": False, "error": f"Primary: {e}, Fallback: {fb_e}"}
            
            # If not using codex (local only mode) and it failed
            logging.error(f"LLM Generation failed: {e}")
            return {"success": False, "error": str(e)}

    def list_local_models(self) -> List[str]:
        """List all models currently installed in Ollama."""
        target_url = self.primary_url if not self.use_codex else self.fallback_url
        try:
            response = self.session.get(f"{target_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logging.error(f"Error listing models: {e}")
            return []
