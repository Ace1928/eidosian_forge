#!/usr/bin/env python3
"""
Interactive quickstart for the Ollama Forge.

This script demonstrates the basic functionality of the Ollama Forge
and helps users get started with using the toolkit.
"""
import sys
from typing import Optional, List


from ollama_forge.client import OllamaClient
from helpers.common import (
        print_header,
        print_success,
        print_error,
        print_info,
        print_warning,
        print_json,
        ensure_ollama_running,
    )
from helpers.model_constants import (
        DEFAULT_CHAT_MODEL,
        BACKUP_CHAT_MODEL,
    )

def check_ollama_status() -> bool:
    """Check if Ollama is running and print status."""
    print_header("Checking Ollama Status")
    
    is_running, message = ensure_ollama_running()
    if is_running:
        print_success(f"Ollama is ready: {message}")
        return True
    else:
        print_error(f"Ollama setup issue: {message}")
        print_info("Please install Ollama from https://ollama.com/download")
        return False


def list_available_models(client: OllamaClient) -> List[str]:
    """List available models and return their names."""
    print_header("Available Models")
    
    try:
        models_response = client.list_models()
        if "models" in models_response:
            models = models_response["models"]
            if not models:
                print_warning("No models found. Consider pulling a model.")
                return []
                
            print_info(f"Found {len(models)} available models:")
            for i, model in enumerate(models, 1):
                name = model.get("name", "Unknown")
                print(f"  {i}. {name}")
                
            return [m.get("name", "") for m in models if "name" in m]
        else:
            print_error("Failed to retrieve models list")
            return []
    except Exception as e:
        print_error(f"Error listing models: {str(e)}")
        return []


def interactive_chat(client: OllamaClient, model: Optional[str] = None) -> None:
    """Run an interactive chat session with a model."""
    print_header("Interactive Chat")
    
    # Use provided model, DEFAULT_CHAT_MODEL, or prompt user
    if not model:
        models = list_available_models(client)
        if models:
            default_model = next((m for m in models if DEFAULT_CHAT_MODEL in m), models[0])
            model_index = input(f"Select model (1-{len(models)}, default={models.index(default_model)+1}): ")
            try:
                if model_index.strip():
                    idx = int(model_index) - 1
                    model = models[idx]
                else:
                    model = default_model
            except (ValueError, IndexError):
                print_warning("Invalid selection, using default model")
                model = default_model
        else:
            model = DEFAULT_CHAT_MODEL
            print_info(f"No models available, will attempt to use {model}")
    
    print_info(f"Starting chat with model: {model}")
    print_info("Type 'exit', 'quit', or press Ctrl+C to end the chat")
    print()
    
    messages = []
    if input("Add a system message? (y/N): ").lower() == 'y':
        system_msg = input("System message: ")
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
    
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ('exit', 'quit'):
                break
                
            messages.append({"role": "user", "content": user_input})
            
            print("\nAssistant: ", end='', flush=True)
            full_response = ""
            
            try:
                for chunk in client.chat(model, messages, stream=True):
                    if "error" in chunk:
                        print_error(f"\nError: {chunk['error']}")
                        break
                    
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        print(content, end='', flush=True)
                        full_response += content
                
                print()  # New line after response
                
                if full_response:  # Only add to messages if we got a response
                    messages.append({"role": "assistant", "content": full_response})
                    
            except Exception as e:
                print_error(f"\nError during chat: {str(e)}")
                
    except KeyboardInterrupt:
        print("\n\nChat session ended.")


def show_quickstart_menu() -> None:
    """Display the interactive quickstart menu."""
    client = OllamaClient()
    
    while True:
        print_header("Ollama Forge Quickstart")
        print("1. Check Ollama Status")
        print("2. List Available Models")
        print("3. Interactive Chat")
        print("4. Pull a Model")
        print("5. Show Example Code")
        print("0. Exit")
        
        choice = input("\nSelect an option (0-5): ")
        
        if choice == '0':
            break
        elif choice == '1':
            check_ollama_status()
        elif choice == '2':
            list_available_models(client)
        elif choice == '3':
            interactive_chat(client)
        elif choice == '4':
            model_name = input("Enter model name to pull (e.g., llama2): ")
            if model_name:
                print_info(f"Pulling model: {model_name}")
                try:
                    for chunk in client.pull_model(model_name, stream=True):
                        if "completed" in chunk:
                            print(f"Progress: {chunk.get('completed', 0)}/{chunk.get('total', 100)}")
                        elif "status" in chunk:
                            print(f"Status: {chunk['status']}")
                    print_success(f"Model {model_name} pulled successfully")
                except Exception as e:
                    print_error(f"Error pulling model: {str(e)}")
        elif choice == '5':
            show_example_code()
        else:
            print_warning("Invalid choice, please try again")
            
        input("\nPress Enter to continue...")


def show_example_code() -> None:
    """Show example code snippets."""
    print_header("Example Code")
    
    print("""
# Basic usage
from ollama_forge import OllamaClient

client = OllamaClient()
response = client.generate("llama2", "Explain quantum computing in simple terms")
print(response["response"])

# Chat with streaming
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell me a joke"}
]
for chunk in client.chat("llama2", messages, stream=True):
    if "message" in chunk and "content" in chunk["message"]:
        print(chunk["message"]["content"], end="", flush=True)

# Create embeddings
embedding = client.create_embedding("llama2", "This is a test sentence")
print(embedding["embedding"])
""")


def main() -> Optional[int]:
    """Main entry point for the quickstart."""
    try:
        check_ollama_status()
        show_quickstart_menu()
        return 0
    except KeyboardInterrupt:
        print("\nQuickstart interrupted.")
        return 0
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
