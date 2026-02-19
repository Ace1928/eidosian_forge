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
FINAL_DOCS_DIR = ROOT_DIR / 'doc_forge' / 'final_docs'
STATUS_FILE = ROOT_DIR / 'doc_forge' / 'forge_status.json'
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

def update_status(stats):
    """Update the status file with current progress."""
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception:
        pass # Don't crash on status update failure

def get_file_content(rel_path):
    abs_path = ROOT_DIR / rel_path
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def call_llm(prompt):
    """Call the local llama-server with the given prompt."""
    import requests
    import json

    def _registry_port(service_key: str, fallback: int) -> int:
        registry = ROOT_DIR / "config" / "ports.json"
        if not registry.exists():
            return fallback
        try:
            payload = json.loads(registry.read_text(encoding="utf-8"))
            services = payload.get("services", {})
            value = int((services.get(service_key) or {}).get("port", fallback))
            return value if value > 0 else fallback
        except Exception:
            return fallback

    default_port = _registry_port("doc_forge_llm", 8093)
    url = os.environ.get("EIDOS_DOC_FORGE_COMPLETION_URL", f"http://127.0.0.1:{default_port}/completion")
    
    # Qwen-style prompt
    full_prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
    
    payload = {
        "prompt": full_prompt,
        "n_predict": 2048,
        "temperature": 0.7,
        "stop": ["<|im_end|>", "<|im_start|>"],
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=600)
        if response.status_code != 200:
            return f"ERROR: llama-server returned status {response.status_code}\n{response.text}"
        
        data = response.json()
        return data.get("content", "").strip()
    except Exception as e:
        return f"ERROR: Failed to connect to llama-server: {e}"

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

def process_file(rel_dir, filename, stats):
    rel_path = Path(rel_dir) / filename
    staging_path = STAGING_DIR / rel_path.with_suffix('.md')
    final_path = FINAL_DOCS_DIR / rel_path.with_suffix('.md')
    
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    
    if staging_path.exists():
        print(f"Skipping {rel_path} - already exists in staging.")
        stats['skipped'] += 1
        return
        
    if final_path.exists():
        print(f"Skipping {rel_path} - already exists in final_docs.")
        stats['skipped'] += 1
        return
        
    print(f"Processing {rel_path}...")
    content = get_file_content(rel_path)
    
    prompt = f"""Please document the following file: {rel_path}

CONTENT:
```
{content}
```"""
    
    max_retries = 2
    for attempt in range(max_retries + 1):
        llm_output = call_llm(prompt)
        passed, msg = pre_check(llm_output)
        
        if passed:
            with open(staging_path, 'w') as f:
                f.write(llm_output)
            print(f"Successfully documented {rel_path} to {staging_path}")
            stats['processed'] += 1
            return
        else:
            print(f"Attempt {attempt + 1} failed for {rel_path}: {msg}")
            if attempt < max_retries:
                time.sleep(2) # Brief wait before retry
    
    stats['errors'] += 1
    print(f"Failed to document {rel_path} after retries.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate documentation using local LLM.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process (0 for all).")
    args = parser.parse_args()

    if not LLAMA_CLI.exists():
        print("Waiting for llama-cli to be built...")
        return
        
    index = load_index()
    
    stats = {
        'total_files': sum(len(files) for files in index.values()),
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'start_time': time.time()
    }
    
    count = 0
    # Process files directory-by-directory
    for rel_dir, files in index.items():
        print(f"Entering directory: {rel_dir}")
        for filename in files:
            if args.limit > 0 and count >= args.limit:
                print(f"Limit of {args.limit} files reached.")
                update_status(stats)
                return
            
            process_file(rel_dir, filename, stats)
            update_status(stats)
            count += 1

if __name__ == "__main__":
    main()
