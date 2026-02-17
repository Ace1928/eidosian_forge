#!/usr/bin/env python3
import os
import json
import subprocess
import time
from pathlib import Path

# Configuration
ROOT_DIR = Path.cwd()
INDEX_FILE = ROOT_DIR / 'doc_forge' / 'file_index.json'
STAGING_DIR = ROOT_DIR / 'doc_forge' / 'staging'
LLAMA_CLI = ROOT_DIR / 'doc_forge' / 'llama.cpp' / 'build' / 'bin' / 'llama-cli'
MODEL_PATH = ROOT_DIR / 'doc_forge' / 'models' / 'qwen2.5-1.5b-instruct-q5_k_m.gguf'

SYSTEM_PROMPT = """You are an Eidosian Documentation Forge agent. Your task is to generate detailed, accurate, and structurally elegant documentation for the provided source file.
Follow these requirements:
1. File Summary: High-level overview of what the file does.
2. API Documentation: List all functions, classes, and their parameters/methods.
3. Current Status: Accurate assessment of the file's current state based on its content and comments.
4. Potential Future Directions: Eidosian ideas for enhancement or refactoring.
Ensure the output is in valid Markdown format. Maintain a professional, precise, and witty Eidosian tone ("Velvet Beef"). Avoid placeholders like "Not provided" or "TODO" in your analysis; instead, describe what is actually present.
"""

def load_index():
    with open(INDEX_FILE, 'r') as f:
        return json.load(f)

def get_file_content(rel_path):
    abs_path = ROOT_DIR / rel_path
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def call_llm(prompt):
    """Call the local llama-cli with the given prompt."""
    if not LLAMA_CLI.exists():
        return "ERROR: llama-cli not found. Build may still be in progress."
    
    # Use a simpler prompt for the CLI if needed, or wrap it in instruct tags
    # Qwen-style prompt
    full_prompt = f"<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"
    
    cmd = [
        str(LLAMA_CLI),
        "-m", str(MODEL_PATH),
        "-p", full_prompt,
        "-n", "1024", # Limit output length
        "--temp", "0.7",
        "--repeat-penalty", "1.1",
        "--no-display-prompt"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return f"ERROR: llama-cli failed with return code {result.returncode}
{result.stderr}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "ERROR: llama-cli timed out after 5 minutes."
    except Exception as e:
        return f"ERROR: Unexpected error calling llama-cli: {e}"

def pre_check(content):
    """Perform automated gating and quality checks on the LLM output."""
    if not content:
        return False, "Empty content"
    if "ERROR:" in content:
        return False, content
    if "TODO" in content or "Not implemented" in content:
        # Check if it's the LLM saying it's not implemented vs the code
        if "I cannot provide" in content or "The file content is missing" in content:
            return False, "LLM failed to analyze properly"
    return True, "OK"

def process_file(rel_dir, filename):
    rel_path = Path(rel_dir) / filename
    staging_path = STAGING_DIR / rel_path.with_suffix('.md')
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    
    if staging_path.exists():
        print(f"Skipping {rel_path} - already exists in staging.")
        return
        
    print(f"Processing {rel_path}...")
    content = get_file_content(rel_path)
    
    prompt = f"Please document the following file: {rel_path}

CONTENT:
```
{content}
```"
    
    max_retries = 2
    for attempt in range(max_retries + 1):
        llm_output = call_llm(prompt)
        passed, msg = pre_check(llm_output)
        
        if passed:
            with open(staging_path, 'w') as f:
                f.write(llm_output)
            print(f"Successfully documented {rel_path} to {staging_path}")
            return
        else:
            print(f"Attempt {attempt + 1} failed for {rel_path}: {msg}")
            if attempt < max_retries:
                time.sleep(2) # Brief wait before retry

def main():
    if not LLAMA_CLI.exists():
        print("Waiting for llama-cli to be built...")
        return
        
    index = load_index()
    
    # Process files directory-by-directory
    for rel_dir, files in index.items():
        print(f"Entering directory: {rel_dir}")
        for filename in files:
            process_file(rel_dir, filename)

if __name__ == "__main__":
    main()
