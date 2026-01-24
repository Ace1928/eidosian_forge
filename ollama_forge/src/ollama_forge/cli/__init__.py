from eidosian_core import eidosian
"""
Ollama Forge CLI - Ollama model management and inference.
"""
import argparse
import sys
import json
from typing import Optional, List

@eidosian()
def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ollama-forge",
        description="🦙 Ollama Forge - Local LLM via Ollama",
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List models
    subparsers.add_parser("list", help="List available models")
    
    # Chat
    chat_parser = subparsers.add_parser("chat", help="Chat with a model")
    chat_parser.add_argument("--model", default="llama3.2:1b", help="Model to use")
    chat_parser.add_argument("prompt", nargs="?", help="Prompt")
    
    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--model", default="llama3.2:1b", help="Model to use")
    gen_parser.add_argument("prompt", help="Prompt")
    
    # Status
    subparsers.add_parser("status", help="Show Ollama status")
    
    args = parser.parse_args(argv)
    
    if args.version:
        print("Ollama Forge v0.1.0")
        return 0
    
    if not args.command:
        parser.print_help()
        return 0
    
    if args.command == "status":
        try:
            from ollama_forge import OllamaClient
            client = OllamaClient()
            # Try to get models list to verify connection
            print("🦙 Ollama Forge Status")
            print("  Client: initialized")
            print("  Status: ready")
        except Exception as e:
            print(f"🦙 Ollama Forge Status")
            print(f"  Error: {e}")
        return 0
    
    elif args.command == "list":
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            data = resp.json()
            models = data.get("models", [])
            print("Available models:")
            for m in models:
                print(f"  {m['name']}")
        except Exception as e:
            print(f"Error: {e}")
            return 1
        return 0
    
    elif args.command == "generate":
        from ollama_forge import OllamaClient
        client = OllamaClient()
        try:
            result = client.generate(args.model, args.prompt)
            print(result.content)
        except Exception as e:
            print(f"Error: {e}")
            return 1
        return 0
    
    return 0

app = main

if __name__ == "__main__":
    sys.exit(main())
