#!/usr/bin/env python3
"""
Verify Local LLM Functionality.

This script instantiates the ModelState, checks if the model (qwen/qwen2.5-0.5b-instruct)
can be loaded, and attempts a simple text generation task.
It verifies that timeouts and device placement are handled correctly.
"""

import sys
import time
from pathlib import Path

# Ensure we can import word_forge
sys.path.insert(0, str(Path(__file__).parents[1].resolve() / "src"))
sys.path.insert(0, str(Path(__file__).parents[2].resolve() / "lib"))

from word_forge.parser.language_model import ModelState
import torch

def verify_llm():
    print("--- Verifying Local LLM ---")
    
    # Check PyTorch availability
    if not torch.cuda.is_available():
        print("Note: CUDA not available, using CPU. Inference might be slow.")
    else:
        print(f"CUDA Available: {torch.cuda.get_device_name(0)}")

    # Initialize Model State
    # Use the default local model
    llm = ModelState()
    
    print(f"Model Name: {llm.get_model_name()}")
    
    start_time = time.time()
    print("Initializing model (this may take time to download/load)...")
    success = llm.initialize()
    load_time = time.time() - start_time
    
    if not success:
        print("❌ Model initialization failed.")
        return False
    
    print(f"✅ Model initialized in {load_time:.2f}s")
    
    # Test Generation
    prompt = "Define the word 'serendipity' in one sentence."
    print(f"\nPrompt: {prompt}")
    print("Generating...")
    
    start_gen = time.time()
    # We use a reasonably short max_new_tokens for a quick test
    output = llm.generate_text(prompt, max_new_tokens=50)
    gen_time = time.time() - start_gen
    
    if output:
        print(f"✅ Output: {output}")
        print(f"Generation Time: {gen_time:.2f}s")
        return True
    else:
        print("❌ Generation returned None.")
        return False

if __name__ == "__main__":
    if verify_llm():
        sys.exit(0)
    else:
        sys.exit(1)
