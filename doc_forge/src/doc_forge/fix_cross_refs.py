#!/usr/bin/env python3
# 🌀 Eidosian Cross-Reference Fixer
"""
Cross-Reference Fixer - Resolving Ambiguities in Documentation References

This script analyzes Python source files and documentation references to resolve
ambiguities in cross-references. It identifies and fixes common problems like
duplicate class references and improper imports.

Following Eidosian principles of:
- Contextual Integrity: References should have clear context
- Precision as Style: References must be exact
- Structure as Control: Proper namespacing enforces meaning
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 📊 Structured Logging - Self-Awareness Foundation
logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s [%(levelname)8s] %(message)s")
logger = logging.getLogger("eidosian_docs.cross_ref_fixer")

def fix_ambiguous_references(repo_root: Path) -> None:
    """
    Fix ambiguous references in documentation by adding namespace qualifiers.
    
    This function:
    1. Identifies API references that might be ambiguous
    2. Adds explicit module paths to resolve ambiguities
    3. Updates references in RST and Python files
    
    Args:
        repo_root: Root directory of the repository
    """
    docs_dir = repo_root / "docs"  # 📁 Documentation home
    autoapi_dir = docs_dir / "autoapi"  # 🤖 Where AutoAPI generates its output
    
    if not autoapi_dir.exists():
        logger.warning("🔍 AutoAPI directory not found. Run Sphinx build first.")
        return
    
    # Step 1: Find all RST files with ambiguous references - hunting for suspects
    ambiguous_files = []
    for rst_file in autoapi_dir.glob("**/*.rst"):
        with open(rst_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Look for "more than one target" warnings in comments or referenced exception classes
        if "more than one target" in content or ":exc:" in content or ":class:" in content and any(err in content for err in ["Error", "Exception"]):
            ambiguous_files.append(rst_file)
            
    logger.info(f"🔎 Found {len(ambiguous_files)} files with potential ambiguous references")
    
    # Step 2: Process each file - fix them one by one
    for rst_file in ambiguous_files:
        fix_file_references(rst_file)
        
def fix_file_references(file_path: Path) -> None:
    """
    Fix ambiguous references in a specific file with surgical precision.
    
    Args:
        file_path: Path to the file to fix
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # List of common exception classes that could be ambiguous
    exception_classes = [
        'ConnectionError',  # 🔌 Network troubles
        'ModelNotFoundError',  # 🧩 Missing model
        'ServerError',  # 🖥️ Server issues
        'InvalidRequestError',  # 📝 Bad request
        'TimeoutError',  # ⏱️ Time's up!
        'OllamaAPIError',  # 🧠 API troubles
        'APIError',  # 🌐 General API woes
        'ValueError',  # ❓ Bad values
        'TypeError',  # 🔢 Wrong types
    ]
    
    # Pattern to find references to exception classes without full qualification
    updated_content = content
    for exception in exception_classes:
        # Pattern 1: Direct exception references - catch them in the wild!
        pattern = rf':(?:class|exc):`({exception})`'
        # Replace with fully qualified names for exceptions only - give them a proper home
        if exception not in ['ValueError', 'TypeError']:  # These are Python builtins - leave them be
            updated_content = re.sub(
                pattern,
                r':class:`ollama_forge.exceptions.\1`',
                updated_content
            )
        
        # Pattern 2: References in :raises: directives - error handling documentation
        pattern = rf':raises\s+({exception}):'
        if exception not in ['ValueError', 'TypeError']:
            updated_content = re.sub(
                pattern,
                r':raises ollama_forge.exceptions.\1:',
                updated_content
            )
    
    # Add :noindex: directive to duplicate references - prevent duplicate object warnings
    pattern = r'(\.\. py:[a-z]+:: [a-zA-Z0-9_.]+\.[a-zA-Z0-9_]+Error)'
    matches = list(re.finditer(pattern, updated_content))
    if len(matches) > 1:  # If we found duplicates
        for match in matches[1:]:  # Keep first instance, mark others as duplicates
            if ':noindex:' not in updated_content[match.start():match.start() + 100]:
                directive = match.group(1)
                replacement = f"{directive}\n   :noindex:"
                updated_content = updated_content[:match.start(1)] + replacement + updated_content[match.end(1):]
    
    # Write the updated content if changed - only make changes when necessary
    if updated_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        logger.info(f"✅ Fixed references in {file_path.name}")

def fix_python_imports(repo_root: Path) -> None:
    """
    Fix Python imports to prevent ambiguous references.
    
    Args:
        repo_root: Root directory of the repository
    """
    source_dir = repo_root / "ollama_forge"
    
    if not source_dir.exists():
        logger.warning("Source directory not found.")
        return
    
    for py_file in source_dir.glob("**/*.py"):
        if py_file.name == "__init__.py":
            # Check for re-exports that cause ambiguity
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Look for imports of exceptions
            if "from .exceptions import" in content:
                # Add noqa comments to prevent ambiguity warnings
                updated_content = re.sub(
                    r'(from \.exceptions import .*?)(?:\n|$)',
                    r'\1  # noqa: F401\n',
                    content
                )
                
                if updated_content != content:
                    with open(py_file, "w", encoding="utf-8") as f:
                        f.write(updated_content)
                    logger.info(f"✅ Added noqa to imports in {py_file.name}")

# Add a new function to create a proper intersphinx configuration
def create_intersphinx_mapping(repo_root: Path) -> None:
    """
    Create or update the intersphinx_mapping in conf.py to help with cross-references.
    
    Args:
        repo_root: Root directory of the repository
    """
    conf_path = repo_root / "docs" / "conf.py"
    if not conf_path.exists():
        logger.warning(f"conf.py not found at {conf_path}")
        return
        
    with open(conf_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Check if intersphinx_mapping already exists
    if "intersphinx_mapping" in content and "sphinx.ext.intersphinx" in content:
        logger.info("intersphinx_mapping already exists in conf.py")
        return
        
    # Add intersphinx extension if needed
    if "sphinx.ext.intersphinx" not in content:
        extensions_pattern = r"(extensions\s*=\s*\[.*?\])"
        extensions_match = re.search(extensions_pattern, content, re.DOTALL)
        if extensions_match:
            extensions_str = extensions_match.group(1)
            if extensions_str.endswith("]"):
                new_extensions = extensions_str[:-1] + ',\n    "sphinx.ext.intersphinx",\n]'
                content = content.replace(extensions_str, new_extensions)
            
    # Add intersphinx_mapping if needed
    if "intersphinx_mapping" not in content:
        mapping = """
# Intersphinx mapping for cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
"""
        # Find a good place to insert the mapping (after extensions)
        if "extensions" in content:
            extensions_end = content.find("]", content.find("extensions"))
            if extensions_end > 0:
                content = content[:extensions_end+1] + "\n" + mapping + content[extensions_end+1:]
        else:
            # Append to end of file if extensions not found
            content += "\n" + mapping
            
    # Write updated content
    with open(conf_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("✅ Added intersphinx configuration to conf.py")

if __name__ == "__main__":
    # Find repository root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    
    logger.info(f"🔍 Scanning for ambiguous references in {repo_root}")
    fix_ambiguous_references(repo_root)
    fix_python_imports(repo_root)
    create_intersphinx_mapping(repo_root)
    
    logger.info("✅ Cross-reference fixing complete")
