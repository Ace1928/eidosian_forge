#!/usr/bin/env python3
"""
Basic usage examples for the Ollama Forge client.

This script provides simple, clean examples of the core functionality
to help users quickly understand how to use the client.
"""
import time
from typing import List, Callable

# Import OllamaClient
from ollama_forge import OllamaClient

# Import constants from model_constants
from helpers.model_constants import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL
)

# Define printer function type for strict typing
PrinterFunc = Callable[[str], None]

from helpers.common import (
    print_header,
    print_success,
    print_error,
    print_info,
    print_warning
)

def basic_client_setup() -> OllamaClient:
    """Basic client initialization with default settings."""
    print_header("1. Basic Client Setup")
    
    # Initialize with defaults
    client = OllamaClient()
    print_success("Client initialized with default settings")
    print_info("Default API URL: http://localhost:11434")
    
    # Custom client is commented out to avoid unused variable warning
    # client_custom = OllamaClient(
    #     base_url="http://localhost:11434",
    #     timeout=60,
    #     max_retries=3
    # )
    # print_success("Custom client created with specific settings")
    
    return client


def check_api_connection(client: OllamaClient) -> bool:
    """Test the API connection and get version information."""
    print_header("2. API Connection Test")
    
    try:
        version_info = client.get_version()
        print_success("Connected to Ollama API")
        print_info(f"Server version: {version_info.get('version', 'unknown')}")
        return True
    except Exception as e:
        print_error(f"Failed to connect to Ollama API: {e}")
        print_warning("Make sure Ollama server is running with: 'ollama serve'")
        return False


def list_available_models(client: OllamaClient) -> List[str]:
    """List available models and their details."""
    print_header("3. Available Models")
    
    try:
        models_response = client.list_models()
        models = models_response.get("models", [])
        
        if not models:
            print_warning("No models found. Pull a model first.")
            print_info("Example: client.pull_model('llama2')")
            return []
        
        # Print model list
        print_success(f"Found {len(models)} models:")
        model_names: List[str] = []
        for i, model in enumerate(models, 1):
            name = model.get("name", "Unknown")
            size = model.get("size", "Unknown size")
            # modified = model.get("modified_at", "Unknown date")
            print(f"  {i}. {name} ({size})")
            model_names.append(name)
            
        return model_names
    except Exception as e:
        print_error(f"Error listing models: {e}")
        return []


def generate_text(client: OllamaClient, model: str = DEFAULT_CHAT_MODEL) -> None:
    """Demonstrate basic text generation."""
    print_header("4. Text Generation")
    
    prompt = "Explain quantum computing in one paragraph."
    print_info(f"Generating text with model: {model}")
    print_info(f"Prompt: '{prompt}'")
    
    # Non-streaming generation
    try:
        start_time = time.time()
        response = client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.7}
        )
        duration = time.time() - start_time
        
        print_success(f"Generation completed in {duration:.2f}s")
        response_dict = response if isinstance(response, dict) else {}
        response_text = response_dict.get("response", "No response")
        print("\nResponse:")
        print(response_text)
        print()
    except Exception as e:
        print_error(f"Generation failed: {e}")


def chat_completion(client: OllamaClient, model: str = DEFAULT_CHAT_MODEL) -> None:
    """Demonstrate chat completion functionality."""
    print_header("5. Chat Completion")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about artificial intelligence."}
    ]
    print_info(f"Sending chat request to model: {model}")
    print_info("Messages:")
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"  {role}: {content}")
    
    try:
        # Streaming chat
        print("\nStreaming response:")
        print("ASSISTANT: ", end="", flush=True)
        
        full_response: str = ""
        for chunk in client.chat(
            model=model,
            messages=messages,
            stream=True
        ):
            if isinstance(chunk, dict) and "message" in chunk:
                message = chunk["message"]
                if isinstance(message, dict) and "content" in message:
                    content = str(message["content"])
                    print(content, end="", flush=True)
                    full_response += content
        
        print("\n")  # End with newline
    except Exception as e:
        print_error(f"\nChat completion failed: {e}")


def create_embeddings(client: OllamaClient, model: str = DEFAULT_EMBEDDING_MODEL) -> None:
    """Demonstrate embedding creation and comparison."""
    print_header("6. Creating Embeddings")
    
    texts = [
        "Artificial intelligence is changing the world",
        "Machine learning is a subset of AI",
        "Cheese is made from milk"
    ]
    
    try:
        embeddings: List[List[float]] = []
        for i, text in enumerate(texts):
            print_info(f"Creating embedding for text {i+1}: '{text}'")
            response = client.create_embedding(model=model, prompt=text)
            embedding: List[float] = response["embedding"]
            embeddings.append(embedding)
            print_success(f"Created embedding with {len(embedding)} dimensions")
        
        # Compare embeddings to demonstrate similarity
        if len(embeddings) >= 3:
            # Define a simple similarity function
            def simple_similarity(v1: List[float], v2: List[float]) -> float:
                """Simple cosine similarity implementation."""
                if len(v1) != len(v2): 
                    return 0.0
                dot = sum(a*b for a, b in zip(v1, v2))
                mag1 = sum(a*a for a in v1) ** 0.5
                mag2 = sum(b*b for b in v2) ** 0.5
                return dot / (mag1 * mag2) if mag1 * mag2 > 0 else 0.0
            
            # Use our simple similarity function by default
            calc_sim: Callable[[List[float], List[float]], float] = simple_similarity
            
            # Try to use the helper function if available
            try:
                from ollama_forge.helpers.embedding import calculate_similarity
                calc_sim = calculate_similarity  # type: Callable[[List[float], List[float]], float]
            except ImportError:
                print_warning("Using fallback similarity calculation")
            
            # Calculate similarities
            sim_1_2: float = calc_sim(embeddings[0], embeddings[1])
            sim_1_3: float = calc_sim(embeddings[0], embeddings[2])
            
            print_info("\nSimilarity comparison:")
            print(f"  AI text vs ML text: {sim_1_2:.4f} (should be high)")
            print(f"  AI text vs cheese text: {sim_1_3:.4f} (should be lower)")
    except Exception as e:
        print_error(f"Embedding creation failed: {e}")


def model_management(client: OllamaClient) -> None:
    """Demonstrate model management operations."""
    print_header("7. Model Management")
    
    print_info("These operations actually modify your model library.")
    print_info("Uncomment the code below to try them.")
    
    '''
    # Pulling a model (commented out to avoid unexpected downloads)
    model_to_pull = "llama2:latest"
    print_info(f"Pulling model: {model_to_pull}")
    
    # Use with caution - this will download models
    for progress in client.pull_model(model_to_pull, stream=True):
        if "status" in progress:
            print(f"Status: {progress['status']}")
        if "completed" in progress and "total" in progress:
            percent = (progress["completed"] / progress["total"]) * 100
            print(f"Progress: {percent:.1f}%")
    '''


def main() -> None:
    """Main function demonstrating basic usage examples."""
    print_header("Ollama Forge - Basic Usage Examples")
    
    try:
        # Initialize client
        client = basic_client_setup()
        
        # Check API connection
        if not check_api_connection(client):
            print_error("Failed to connect to Ollama API. Exiting.")
            return
        
        # Run examples
        models = list_available_models(client)
        # Choose first model if available, otherwise use default
        model_name: str = models[0] if models else DEFAULT_CHAT_MODEL
        
        generate_text(client, model_name)
        chat_completion(client, model_name)
        create_embeddings(client, model_name)
        model_management(client)
        
        print_header("All Examples Complete")
        print_success("Successfully demonstrated basic usage of Ollama Forge")
        print_info("Check out other examples in the examples/ directory")
        
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
