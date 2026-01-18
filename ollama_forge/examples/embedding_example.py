#!/usr/bin/env python3
"""
Example script to create and manipulate embeddings using the Ollama API.
"""

# Standard library imports first
from typing import Dict, List, Optional, Tuple, Any
import argparse
import json
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored terminal output
init()

from ollama_forge import OllamaClient
from helpers.common import print_header, print_success, print_error, print_info
from helpers.embedding import (
    calculate_similarity, normalize_vector, batch_calculate_similarities
)
from helpers.model_constants import DEFAULT_EMBEDDING_MODEL, BACKUP_EMBEDDING_MODEL

def create_embedding(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    use_fallback: bool = True
) -> Dict[str, Any]:
    """
    Create an embedding for the given text.
    
    Args:
        text: Input text to embed
        model: Model name to use
        use_fallback: Whether to try fallback model on failure
        
    Returns:
        Dictionary containing the embedding
    """
    client = OllamaClient()
    
    try:
        response = client.create_embedding(model=model, prompt=text)
        if "embedding" not in response and "embeddings" in response:
            # Handle different response formats
            response["embedding"] = response["embeddings"][0] if isinstance(response["embeddings"], list) else response["embeddings"]
            
        return response
    except Exception as e:
        print_error(f"Error creating embedding with {model}: {e}")
        
        if use_fallback and model != BACKUP_EMBEDDING_MODEL:
            print_info(f"Trying fallback model {BACKUP_EMBEDDING_MODEL}...")
            return create_embedding(text, BACKUP_EMBEDDING_MODEL, use_fallback=False)
        
        raise e

def find_similar_texts(
    query: str,
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 3
) -> List[Tuple[int, float, str]]:
    """
    Find texts similar to the query text using embedding similarity.
    
    Args:
        query: The query text
        texts: List of texts to compare against
        model: Model name for embeddings
        top_k: Number of top results to return
        
    Returns:
        List of (index, similarity, text) tuples
    """
    # Create embeddings
    query_embedding = create_embedding(query, model)["embedding"]
    
    # Create embeddings for all texts
    text_embeddings = []
    for text in texts:
        try:
            embedding = create_embedding(text, model)["embedding"]
            text_embeddings.append(embedding)
        except Exception as e:
            print_error(f"Error creating embedding for text: {e}")
            text_embeddings.append([0.0] * len(query_embedding))  # Zero embedding as fallback
    
    # Calculate similarities
    similarities = batch_calculate_similarities(query_embedding, text_embeddings)
    
    # Get top_k results
    top_results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    # Return (index, similarity, text) tuples
    return [(idx, sim, texts[idx]) for idx, sim in top_results]

def display_vector(vector: List[float], max_items: int = 5) -> str:
    """Format a vector for display, showing only a few items."""
    if len(vector) <= max_items:
        return str(vector)
    else:
        return f"[{', '.join(f'{x:.4f}' for x in vector[:max_items])}, ... ({len(vector)-max_items} more)]"

def main() -> None:
    """Main function demonstrating embedding functionality."""
    print_header("Ollama Embeddings Example")
    
    parser = argparse.ArgumentParser(description="Create and manipulate text embeddings with Ollama")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help=f"Model name (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--compare", action="store_true", help="Compare two texts")
    parser.add_argument("--search", action="store_true", help="Search for similar texts")
    parser.add_argument("--save", help="Save embeddings to file")
    parser.add_argument("--load", help="Load embeddings from file")
    
    args = parser.parse_args()
    
    client = OllamaClient()
    
    # Simple embedding demonstration
    if not (args.compare or args.search or args.save or args.load):
        print_info("Creating a simple embedding...")
        
        text = input("Enter text to embed: ") or "This is a sample text for embedding."
        
        try:
            response = create_embedding(text, args.model)
            embedding = response["embedding"]
            
            print_success(f"Embedding created with {args.model}")
            print(f"Dimensions: {len(embedding)}")
            print(f"Embedding vector: {display_vector(embedding)}")
            
        except Exception as e:
            print_error(f"Failed to create embedding: {e}")
    
    # Compare two texts
    if args.compare:
        print_info("Comparing two texts using embeddings...")
        
        text1 = input("Enter first text: ") or "Artificial intelligence is changing the world."
        text2 = input("Enter second text: ") or "AI is transforming global technology."
        
        try:
            embedding1 = create_embedding(text1, args.model)["embedding"]
            embedding2 = create_embedding(text2, args.model)["embedding"]
            
            similarity = calculate_similarity(embedding1, embedding2)
            
            print_success(f"Similarity: {similarity:.4f}")
            print("Interpretation:")
            if similarity > 0.9:
                print(f"{Fore.GREEN}Very similar texts{Style.RESET_ALL}")
            elif similarity > 0.7:
                print(f"{Fore.CYAN}Moderately similar texts{Style.RESET_ALL}")
            elif similarity > 0.5:
                print(f"{Fore.YELLOW}Somewhat similar texts{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Not very similar texts{Style.RESET_ALL}")
                
        except Exception as e:
            print_error(f"Comparison failed: {e}")
    
    # Semantic search
    if args.search:
        print_info("Semantic search demonstration...")
        
        # Sample corpus - in a real application, this could be loaded from a database
        corpus = [
            "Artificial intelligence is revolutionizing technology.",
            "Machine learning models require significant computational resources.",
            "Natural language processing helps computers understand human language.",
            "Neural networks are inspired by the human brain.",
            "Ethical considerations are important in AI development.",
            "Data privacy concerns continue to grow in the digital age.",
            "Quantum computing may accelerate certain AI algorithms.",
            "Reinforcement learning enables agents to learn through trial and error."
        ]
        
        print("Available texts:")
        for i, text in enumerate(corpus):
            print(f"{i+1}. {text}")
            
        query = input("\nEnter search query: ") or "How does AI relate to human cognition?"
        
        try:
            similar_texts = find_similar_texts(query, corpus, args.model)
            
            print_success("\nTop similar texts:")
            for i, (idx, similarity, text) in enumerate(similar_texts):
                color = Fore.GREEN if similarity > 0.7 else (Fore.YELLOW if similarity > 0.5 else Fore.RED)
                print(f"{i+1}. {color}[{similarity:.4f}]{Style.RESET_ALL} {text}")
                
        except Exception as e:
            print_error(f"Search failed: {e}")
    
    # Save embeddings
    if args.save:
        print_info(f"Saving embeddings to {args.save}...")
        
        texts = [
            "Artificial intelligence",
            "Machine learning",
            "Neural networks",
            "Natural language processing",
            "Computer vision"
        ]
        
        try:
            embeddings = {}
            for text in texts:
                embedding = create_embedding(text, args.model)["embedding"]
                embeddings[text] = embedding
                
            with open(args.save, 'w') as f:
                json.dump(embeddings, f)
                
            print_success(f"Saved {len(embeddings)} embeddings to {args.save}")
            
        except Exception as e:
            print_error(f"Failed to save embeddings: {e}")
    
    # Load embeddings
    if args.load:
        print_info(f"Loading embeddings from {args.load}...")
        
        try:
            with open(args.load, 'r') as f:
                embeddings = json.load(f)
                
            print_success(f"Loaded {len(embeddings)} embeddings")
            
            for text, embedding in list(embeddings.items())[:3]:
                print(f"• {text}: {display_vector(embedding)}")
                
            if len(embeddings) > 3:
                print(f"... and {len(embeddings) - 3} more")
                
        except Exception as e:
            print_error(f"Failed to load embeddings: {e}")

if __name__ == "__main__":
    main()
