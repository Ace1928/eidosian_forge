#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

# Configuration
ROOT_DIR = Path.cwd()
STAGING_DIR = ROOT_DIR / 'doc_forge' / 'staging'
FINAL_DOCS_DIR = ROOT_DIR / 'doc_forge' / 'final_docs'

def check_quality(content):
    """Check if the documentation meets basic quality standards."""
    if not content.strip():
        return False, "Empty content"
    
    required_sections = ["File Summary", "API Documentation", "Current Status"]
    for section in required_sections:
        if section not in content:
            return False, f"Missing section: {section}"
            
    if "ERROR:" in content:
        return False, "Contains error message"
        
    return True

def auto_approve():
    """Automatically approve files that meet quality standards."""
    if not STAGING_DIR.exists():
        print("Staging directory not found.")
        return
        
    approved_count = 0
    rejected_count = 0
    
    for root, dirs, files in os.walk(STAGING_DIR):
        root_path = Path(root)
        try:
            rel_path = root_path.relative_to(STAGING_DIR)
        except ValueError:
            rel_path = Path('.')
        
        for file in files:
            if not file.endswith('.md'):
                continue
                
            staging_file = root_path / file
            final_file = FINAL_DOCS_DIR / rel_path / file
            
            try:
                content = staging_file.read_text(encoding='utf-8')
                passed = check_quality(content)
                
                if passed:
                    final_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(staging_file), str(final_file))
                    print(f"APPROVED: {rel_path / file}")
                    approved_count += 1
                else:
                    print(f"REJECTED: {rel_path / file} (Quality check failed)")
                    rejected_count += 1
            except Exception as e:
                print(f"ERROR processing {staging_file}: {e}")
                
    print(f"\nAuto-approval complete. Approved: {approved_count}, Rejected: {rejected_count}")

if __name__ == "__main__":
    auto_approve()
