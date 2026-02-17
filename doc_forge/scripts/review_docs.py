#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

# Configuration
ROOT_DIR = Path.cwd()
STAGING_DIR = ROOT_DIR / 'doc_forge' / 'staging'
FINAL_DOCS_DIR = ROOT_DIR / 'doc_forge' / 'final_docs'

def list_staging():
    """List all documentation files waiting in staging."""
    staging_files = []
    for root, dirs, files in os.walk(STAGING_DIR):
        root_path = Path(root)
        for file in files:
            if file.endswith('.md'):
                staging_files.append((root_path / file).relative_to(STAGING_DIR))
    return staging_files

def review_file(rel_path):
    staging_file = STAGING_DIR / rel_path
    final_file = FINAL_DOCS_DIR / rel_path
    
    with open(staging_file, 'r') as f:
        print(f"
--- REVIEWING: {rel_path} ---")
        print(f.read())
        print("-" * 30)
    
    choice = input(f"Approve documentation for {rel_path}? (y/n/skip): ").strip().lower()
    
    if choice == 'y':
        final_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging_file), str(final_file))
        print(f"Approved and moved to {final_file}")
    elif choice == 'n':
        print(f"Rejected. Keeping in staging for revision.")
    else:
        print(f"Skipping.")

def main():
    if not STAGING_DIR.exists():
        print("Staging directory not found.")
        return
        
    staging_files = list_staging()
    if not staging_files:
        print("No files in staging to review.")
        return
        
    print(f"Found {len(staging_files)} files in staging.")
    for f in staging_files:
        review_file(f)

if __name__ == "__main__":
    main()
