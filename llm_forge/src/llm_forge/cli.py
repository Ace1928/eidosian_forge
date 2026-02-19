#!/usr/bin/env python3
"""
LLM Forge CLI - Command-line interface for LLM operations.

Standalone Usage:
    llm-forge status              # Show LLM status
    llm-forge models              # List available models
    llm-forge chat <prompt>       # Chat with LLM
    llm-forge embed <text>        # Generate embeddings
    llm-forge compare             # Compare models

Enhanced with other forges:
    - memory_forge: Conversational memory
    - knowledge_forge: RAG-enhanced responses
"""
from __future__ import annotations
from eidosian_core import eidosian
from eidosian_core.ports import get_service_url

import sys
from pathlib import Path
from typing import Optional

# Add lib to path for CLI framework
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "lib"))

from cli import StandardCLI, CommandResult, ForgeDetector

from llm_forge import ModelManager, OllamaProvider

DEFAULT_OLLAMA_URL = get_service_url("ollama_http", default_port=11434, default_host="localhost", default_path="")
DEFAULT_OLLAMA_TAGS_URL = f"{DEFAULT_OLLAMA_URL}/api/tags"
DEFAULT_OLLAMA_EMBED_URL = f"{DEFAULT_OLLAMA_URL}/api/embeddings"


class LLMForgeCLI(StandardCLI):
    """CLI for LLM Forge - unified LLM interface."""
    
    name = "llm_forge"
    description = "Unified LLM interface with caching and comparison capabilities"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self._manager: Optional[ModelManager] = None
        self._ollama: Optional[OllamaProvider] = None
    
    @property
    def manager(self) -> ModelManager:
        """Lazy-load model manager."""
        if self._manager is None:
            self._manager = ModelManager()
        return self._manager
    
    @property
    def ollama(self) -> OllamaProvider:
        """Lazy-load Ollama provider."""
        if self._ollama is None:
            self._ollama = self.manager.ollama_provider
        return self._ollama
    
    @eidosian()
    def register_commands(self, subparsers) -> None:
        """Register llm-forge specific commands."""
        
        # Models command
        models_parser = subparsers.add_parser(
            "models",
            help="List available models",
        )
        models_parser.set_defaults(func=self._cmd_models)
        
        # Chat command
        chat_parser = subparsers.add_parser(
            "chat",
            help="Chat with LLM",
        )
        chat_parser.add_argument(
            "prompt",
            help="Prompt to send",
        )
        chat_parser.add_argument(
            "-m", "--model",
            help="Model to use (default: phi3:mini)",
        )
        chat_parser.add_argument(
            "-t", "--temperature",
            type=float,
            default=0.7,
            help="Temperature (default: 0.7)",
        )
        chat_parser.add_argument(
            "--max-tokens",
            type=int,
            help="Maximum tokens to generate",
        )
        chat_parser.set_defaults(func=self._cmd_chat)
        
        # Embed command
        embed_parser = subparsers.add_parser(
            "embed",
            help="Generate embeddings",
        )
        embed_parser.add_argument(
            "text",
            help="Text to embed",
        )
        embed_parser.add_argument(
            "-m", "--model",
            default="nomic-embed-text",
            help="Embedding model (default: nomic-embed-text)",
        )
        embed_parser.set_defaults(func=self._cmd_embed)
        
        # Config command
        config_parser = subparsers.add_parser(
            "config",
            help="Show model configuration",
        )
        config_parser.set_defaults(func=self._cmd_config)
        
        # Test command
        test_parser = subparsers.add_parser(
            "test",
            help="Test LLM connection",
        )
        test_parser.add_argument(
            "-m", "--model",
            help="Model to test",
        )
        test_parser.set_defaults(func=self._cmd_test)
    
    @eidosian()
    def cmd_status(self, args) -> CommandResult:
        """Show LLM forge status."""
        try:
            # Check Ollama
            ollama_ok = False
            models = []
            try:
                import requests
                resp = requests.get(DEFAULT_OLLAMA_TAGS_URL, timeout=5)
                if resp.status_code == 200:
                    ollama_ok = True
                    data = resp.json()
                    models = [m["name"] for m in data.get("models", [])]
            except Exception:
                pass
            
            integrations = []
            if ForgeDetector.is_available("memory_forge"):
                integrations.append("memory_forge")
            if ForgeDetector.is_available("knowledge_forge"):
                integrations.append("knowledge_forge")
            
            return CommandResult(
                True,
                f"LLM Forge operational - Ollama {'connected' if ollama_ok else 'disconnected'}",
                {
                    "ollama_connected": ollama_ok,
                    "models_available": len(models),
                    "models": models[:10],
                    "integrations": integrations,
                }
            )
        except Exception as e:
            return CommandResult(False, f"Error: {e}")
    
    def _cmd_models(self, args) -> None:
        """List available models."""
        try:
            import requests
            resp = requests.get(DEFAULT_OLLAMA_TAGS_URL, timeout=10)
            
            if resp.status_code != 200:
                result = CommandResult(False, "Failed to connect to Ollama")
            else:
                data = resp.json()
                models = []
                for m in data.get("models", []):
                    size_gb = m.get("size", 0) / (1024**3)
                    models.append({
                        "name": m["name"],
                        "size": f"{size_gb:.1f}GB",
                        "modified": m.get("modified_at", "")[:10],
                    })
                
                result = CommandResult(
                    True,
                    f"Found {len(models)} models",
                    {"models": models}
                )
        except Exception as e:
            result = CommandResult(False, f"Error: {e}")
        self._output(result, args)
    
    def _cmd_chat(self, args) -> None:
        """Chat with LLM."""
        try:
            model = args.model or "phi3:mini"
            
            response = self.ollama.generate(
                prompt=args.prompt,
                model=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            
            result = CommandResult(
                True,
                response.content,
                {
                    "model": model,
                    "prompt": args.prompt,
                    "response": response.content,
                    "usage": response.usage if hasattr(response, "usage") else None,
                }
            )
        except Exception as e:
            result = CommandResult(False, f"Chat error: {e}")
        self._output(result, args)
    
    def _cmd_embed(self, args) -> None:
        """Generate embeddings."""
        try:
            import requests
            
            resp = requests.post(
                DEFAULT_OLLAMA_EMBED_URL,
                json={"model": args.model, "prompt": args.text},
                timeout=30,
            )
            
            if resp.status_code != 200:
                result = CommandResult(False, f"Embedding error: {resp.text}")
            else:
                data = resp.json()
                embedding = data.get("embedding", [])
                
                result = CommandResult(
                    True,
                    f"Generated {len(embedding)}-dimensional embedding",
                    {
                        "model": args.model,
                        "dimensions": len(embedding),
                        "embedding_preview": embedding[:5] if embedding else [],
                    }
                )
        except Exception as e:
            result = CommandResult(False, f"Embed error: {e}")
        self._output(result, args)
    
    def _cmd_config(self, args) -> None:
        """Show model configuration."""
        try:
            try:
                from eidos_mcp.config.models import model_config
                config = {
                    "inference_model": model_config.inference.model,
                    "embedding_model": model_config.embedding.model,
                    "embedding_dims": model_config.embedding.dimensions,
                    "ollama_url": model_config.ollama.base_url,
                }
            except ImportError:
                config = {
                    "inference_model": "phi3:mini",
                    "embedding_model": "nomic-embed-text",
                    "embedding_dims": 768,
                    "ollama_url": DEFAULT_OLLAMA_URL,
                    "note": "Using defaults (eidos_mcp not available)",
                }
            
            result = CommandResult(
                True,
                f"Inference: {config['inference_model']}, Embedding: {config['embedding_model']}",
                config
            )
        except Exception as e:
            result = CommandResult(False, f"Error: {e}")
        self._output(result, args)
    
    def _cmd_test(self, args) -> None:
        """Test LLM connection."""
        try:
            model = args.model or "phi3:mini"
            
            response = self.ollama.generate(
                prompt="Say 'hello' in one word.",
                model=model,
                temperature=0.1,
                max_tokens=10,
            )
            
            result = CommandResult(
                True,
                f"Connection OK - Model responded: {response.content[:50]}",
                {
                    "model": model,
                    "response": response.content,
                    "test": "passed",
                }
            )
        except Exception as e:
            result = CommandResult(False, f"Test failed: {e}")
        self._output(result, args)


@eidosian()
def main():
    """Entry point for llm-forge CLI."""
    cli = LLMForgeCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
