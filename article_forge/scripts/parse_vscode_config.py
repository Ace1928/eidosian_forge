#!/usr/bin/env python3
"""
Parses vscode_setup.md to extract VSCode settings (JSON) and extension IDs.
"""
import re
import json
import sys
from pathlib import Path

def parse_vscode_config(markdown_path: Path):
    content = markdown_path.read_text(encoding="utf-8")
    
    # Regex to find JSON blocks
    json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
    
    # Merge all JSON blocks into a single dictionary
    settings = {}
    for block in json_blocks:
        try:
            settings.update(json.loads(block))
        except json.JSONDecodeError as e:
            print(f"Warning: Could not decode JSON block: {e}\n{block[:100]}...", file=sys.stderr)
            
    # Regex to find extension IDs (e.g., "esbenp.prettier-vscode")
    # This looks for patterns like 'extension_id": "publisher.extension-name"' or 'code --install-extension publisher.extension-name'
    extension_pattern = r'(?P<publisher>[a-zA-Z0-9_-]+)\.(?P<name>[a-zA-Z0-9_-]+)'
    
    # Find all occurrences that look like extension IDs
    raw_extension_ids = re.findall(extension_pattern, content)
    
    # Filter for unique and valid-looking extension IDs (e.g., publisher.extension-name)
    extensions = sorted(list(set([f"{p}.{n}" for p, n in raw_extension_ids if p and n])))

    return {"settings": settings, "extensions": extensions}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: parse_vscode_config.py <path_to_vscode_setup.md>")
        sys.exit(1)
        
    md_path = Path(sys.argv[1])
    if not md_path.exists():
        print(f"Error: File not found at {md_path}")
        sys.exit(1)
        
    result = parse_vscode_config(md_path)
    print(json.dumps(result, indent=2))
