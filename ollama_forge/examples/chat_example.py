#!/usr/bin/env python3
"""
Example script for chat interactions with Ollama models.

This example demonstrates various chat capabilities including:
- Basic chat requests
- Streaming responses
- Conversation memory
- System prompts
- Error handling with fallbacks
"""
from typing import Dict, List, Any, Optional, Tuple
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal output
init()

from ollama_forge import OllamaClient
from helpers.common import print_header, print_error, print_info, print_warning
from helpers.model_constants import DEFAULT_CHAT_MODEL, BACKUP_CHAT_MODEL


def initialize_chat(model: str, system_message: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Initialize a chat conversation with optional system message.
    
    Args:
        model: Model to use for chat
        system_message: Optional system message to set behavior
        
    Returns:
        Initial messages list
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    return messages


def chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    use_fallback: bool = True
) -> Dict[str, Any]:
    """
    Send a non-streaming chat request to the model.
    
    Args:
        model: Name of the model to use
        messages: List of message dictionaries (role, content)
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        use_fallback: Whether to try fallback model on failure
        
    Returns:
        API response containing the model's reply
    """
    client = OllamaClient()
    options = {"temperature": temperature}
    if max_tokens:
        options["max_tokens"] = max_tokens
    
    try:
        return client.chat(
            model=model,
            messages=messages.copy(),  # Use copy to avoid modifying original
            stream=False,
            options=options
        )
    except Exception as e:
        print_error(f"Error in chat with {model}: {e}")
        
        if use_fallback and model != BACKUP_CHAT_MODEL:
            print_warning(f"Trying fallback model {BACKUP_CHAT_MODEL}")
            return chat(
                model=BACKUP_CHAT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                use_fallback=False
            )
        
        raise


def chat_streaming(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    use_fallback: bool = True
) -> Tuple[bool, Optional[Dict[str, str]]]:
    """
    Send a streaming chat request and collect the full response.
    
    Args:
        model: Name of model to use
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        use_fallback: Whether to try fallback model on failure
        
    Returns:
        Tuple of (success, complete_message) where the message is a dict with role and content
    """
    client = OllamaClient()
    options = {"temperature": temperature}
    if max_tokens:
        options["max_tokens"] = max_tokens
    
    try:
        # Create a complete assistant message by accumulating streaming chunks
        full_content = ""
        for chunk in client.chat(
            model=model,
            messages=messages.copy(),  # Use copy to avoid modifying original
            stream=True,
            options=options
        ):
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                full_content += content
        
        if not full_content:
            print_warning("Empty response received")
            return False, None
            
        complete_message = {"role": "assistant", "content": full_content}
        return True, complete_message
        
    except Exception as e:
        print_error(f"Error in streaming chat with {model}: {e}")
        
        if use_fallback and model != BACKUP_CHAT_MODEL:
            print_warning(f"Trying fallback model {BACKUP_CHAT_MODEL}")
            return chat_streaming(
                model=BACKUP_CHAT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                use_fallback=False
            )
        
        return False, None


def interactive_chat(
    model: str = DEFAULT_CHAT_MODEL,
    system_message: str = "You are a helpful assistant that provides accurate, thoughtful responses.",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    use_fallback: bool = True
) -> None:
    """
    Run an interactive chat session in the terminal.
    
    Args:
        model: Model name to use
        system_message: System prompt to define assistant behavior
        temperature: Response temperature (0.0 to 1.0)
        max_tokens: Maximum response length in tokens
        use_fallback: Whether to use fallback model on error
    """
    print_header(f"Interactive Chat with {model}")
    print_info(f"System: {system_message}")
    print_info("Type 'exit' or 'quit' to end the session.")
    print_info("Type 'clear' to start a new conversation.")
    print()

    # Initialize conversation with system message
    messages = initialize_chat(model, system_message)
    client = OllamaClient()
    
    while True:
        try:
            # Get user input
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
            if user_input.lower() in ["exit", "quit"]:
                break
            elif user_input.lower() == "clear":
                messages = initialize_chat(model, system_message)
                print_info("Conversation cleared.")
                continue
            elif not user_input.strip():
                continue
            
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Print assistant response with streaming
            print(f"{Fore.BLUE}Assistant: {Style.RESET_ALL}", end="", flush=True)
            assistant_message = {"role": "assistant", "content": ""}
            
            try:
                options = {"temperature": temperature}
                if max_tokens:
                    options["max_tokens"] = max_tokens
                    
                for chunk in client.chat(
                    model=model,
                    messages=messages,
                    stream=True,
                    options=options
                ):
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        print(content, end="", flush=True)
                        assistant_message["content"] += content
                
                # Add assistant response to message history
                messages.append(assistant_message)
                print("\n")
                
            except Exception as e:
                print_error(f"\nError: {e}")
                if use_fallback and model != BACKUP_CHAT_MODEL:
                    print_warning(f"Trying fallback model {BACKUP_CHAT_MODEL}...")
                    
                    try:
                        for chunk in client.chat(
                            model=BACKUP_CHAT_MODEL,
                            messages=messages,
                            stream=True,
                            options=options
                        ):
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                print(content, end="", flush=True)
                                assistant_message["content"] += content
                        
                        # Add assistant response to message history
                        messages.append(assistant_message)
                        print("\n")
                        
                    except Exception as fallback_error:
                        print_error(f"\nFallback error: {fallback_error}")
                        print_warning("Failed to get response. Continuing conversation...")
        
        except KeyboardInterrupt:
            print("\n\nExiting chat...")
            break


def main():
    """Main function demonstrating chat examples."""
    print_header("Ollama Chat Examples")
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Ollama chat examples")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL, help=f"Model name (default: {DEFAULT_CHAT_MODEL})")
    parser.add_argument("--system", default="You are a helpful assistant.", help="System message")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (0.0 to 1.0)")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument("--no-interactive", action="store_true", help="Run non-interactive examples")
    args = parser.parse_args()
    
    # Non-interactive examples
    if args.no_interactive:
        # Example 1: Simple question and answer
        print_info("Example 1: Simple Q&A")
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": "What are three interesting facts about AI?"}
        ]
        
        try:
            response = chat(args.model, messages, args.temperature, args.max_tokens)
            print(f"\nResponse:\n{response['message']['content']}\n")
        except Exception as e:
            print_error(f"Error: {e}")
        
        # Example 2: Multi-turn conversation
        print_info("Example 2: Multi-turn conversation")
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": "I need help planning a vegetarian dinner."},
            {"role": "assistant", "content": "I'd be happy to help you plan a vegetarian dinner! What kind of cuisine are you interested in?"},
            {"role": "user", "content": "Italian cuisine."}
        ]
        
        try:
            response = chat(args.model, messages, args.temperature, args.max_tokens)
            print(f"\nResponse:\n{response['message']['content']}\n")
        except Exception as e:
            print_error(f"Error: {e}")
    
    # Interactive chat
    else:
        interactive_chat(
            model=args.model,
            system_message=args.system,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )


if __name__ == "__main__":
    main()
