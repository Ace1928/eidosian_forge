#!/usr/bin/env python3
"""
Example script for generating text using the Ollama API.

This example demonstrates both streaming and non-streaming text generation,
with various parameter configurations and proper error handling.
"""
import time
from typing import Optional

from ollama_forge import OllamaClient
from helpers.common import (
    print_header, print_success, print_error, print_info, print_warning
)
from helpers.model_constants import DEFAULT_CHAT_MODEL, BACKUP_CHAT_MODEL

def generate_text(
    prompt: str,
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
    stream: bool = False
) -> Optional[str]:
    """
    Generate text using the Ollama API.
    
    Args:
        prompt: The input prompt to generate from
        model: Model name to use
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        stream: Whether to stream the output
        
    Returns:
        Generated text, or None if generation failed
    """
    client = OllamaClient()
    options = {
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        if stream:
            print_info(f"Generating with {model} (streaming)...")
            print("\nResponse:\n")
            
            full_response = ""
            for chunk in client.generate(
                model=model,
                prompt=prompt,
                stream=True,
                options=options
            ):
                if "response" in chunk:
                    text_chunk = chunk["response"]
                    print(text_chunk, end="", flush=True)
                    full_response += text_chunk
                    
            print("\n")  # Add newline at the end
            return full_response
            
        else:
            print_info(f"Generating with {model} (non-streaming)...")
            
            start_time = time.time()
            response = client.generate(
                model=model,
                prompt=prompt,
                stream=False,
                options=options
            )
            elapsed = time.time() - start_time
            
            print("\nResponse:\n")
            print(response["response"])
            print(f"\nGeneration took {elapsed:.2f} seconds")
            
            return response["response"]
            
    except Exception as e:
        print_error(f"Error generating text: {e}")
        
        # Try with fallback model
        if model != BACKUP_CHAT_MODEL:
            print_warning(f"Trying with fallback model {BACKUP_CHAT_MODEL}...")
            return generate_text(
                prompt=prompt,
                model=BACKUP_CHAT_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
        return None

def main() -> None:
    """Main function demonstrating text generation examples."""
    print_header("Ollama Text Generation Example")
    
    # Example prompts
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
        "What are the ethical considerations of AI development?"
    ]
    
    # Get selected prompt
    print("Select a prompt or enter your own:")
    for i, prompt in enumerate(prompts):
        print(f"{i+1}. {prompt}")
    print("4. Enter custom prompt")
    
    choice = input("\nYour choice (1-4): ")
    
    if choice == "4":
        prompt = input("\nEnter your prompt: ")
    else:
        try:
            prompt = prompts[int(choice) - 1]
        except (ValueError, IndexError):
            print_error("Invalid choice, using the first prompt.")
            prompt = prompts[0]
    
    # Get generation parameters
    print("\nGeneration Parameters:")
    model = input(f"Model name (press Enter for {DEFAULT_CHAT_MODEL}): ").strip() or DEFAULT_CHAT_MODEL
    temp_input = input("Temperature (0.0-1.0, press Enter for 0.7): ").strip() or "0.7"
    tokens_input = input("Max tokens (press Enter for 256): ").strip() or "256"
    stream_input = input("Stream output? (y/N): ").strip().lower()
    
    # Parse parameters with error handling
    try:
        temperature = float(temp_input)
        temperature = max(0.0, min(1.0, temperature))  # Clamp between 0 and 1
    except ValueError:
        print_warning("Invalid temperature, using default (0.7)")
        temperature = 0.7
        
    try:
        max_tokens = int(tokens_input)
        max_tokens = max(1, max_tokens)  # Ensure positive
    except ValueError:
        print_warning("Invalid max tokens, using default (256)")
        max_tokens = 256
    
    stream = stream_input == "y"
    
    # Generate text
    result = generate_text(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream
    )
    
    if result:
        print_success("Generation completed successfully!")
    else:
        print_error("Generation failed.")
        
if __name__ == "__main__":
    main()
