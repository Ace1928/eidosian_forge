#!/usr/bin/env python3
# 🌀 Eidosian Doc Forge Initializer
"""
Doc Forge Initializer - Universal Documentation Setup

This script sets up the Doc Forge system for a new or existing project,
creating the necessary directory structure and configuration files.

Following Eidosian principles of:
- Structure as Control: Creating perfect documentation architecture
- Flow Like a River: Seamless setup process
- Self-Awareness: Understanding the project environment
"""

import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("doc_forge.initializer")

def find_repo_root() -> Path:
    """Find the repository root directory."""
    # Start with current directory and move up
    current_dir = Path.cwd()
    
    # Check for common repo indicators
    for directory in [current_dir] + list(current_dir.parents):
        # Check for .git or other repo indicators
        if any((directory / indicator).exists() for indicator in [".git", ".hg", ".svn"]):
            return directory
            
        # Look for common project files
        if (directory / "setup.py").exists() or (directory / "pyproject.toml").exists():
            return directory
    
    # If no repo root found, use current directory
    logger.warning("⚠️ Could not find repository root, using current directory")
    return current_dir

def setup_documentation_structure(repo_root: Path, force: bool = False) -> None:
    """Set up the documentation structure."""
    # Create main directories
    docs_dir = repo_root / "docs"
    dirs_to_create = [
        docs_dir,
        docs_dir / "_static",
        docs_dir / "_static" / "css",
        docs_dir / "_static" / "js",
        docs_dir / "_static" / "img",
        docs_dir / "_templates",
        docs_dir / "user_docs",
        docs_dir / "auto_docs",
        docs_dir / "ai_docs",
        docs_dir / "assets",
    ]
    
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {directory.relative_to(repo_root)}")
    
    # Create basic files
    files_to_create = [
        (docs_dir / "index.md", create_index_content(repo_root.name)),
        (docs_dir / "requirements.txt", create_requirements_content()),
        (docs_dir / "_static" / "css" / "custom.css", create_css_content()),
        (docs_dir / "docs_manifest.json", create_manifest_content(repo_root.name)),
    ]
    
    for file_path, content in files_to_create:
        if not file_path.exists() or force:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"📄 Created file: {file_path.relative_to(repo_root)}")
        else:
            logger.info(f"📄 File already exists (skipping): {file_path.relative_to(repo_root)}")
    
    # Copy conf.py if it doesn't exist
    conf_py = docs_dir / "conf.py"
    if not conf_py.exists() or force:
        # Try to find the template conf.py
        template_paths = [
            repo_root / "src" / "doc_forge" / "templates" / "conf.py.template",
            repo_root / "src" / "templates" / "conf.py.template",
            repo_root / "templates" / "conf.py.template",
        ]
        
        template_found = False
        for template_path in template_paths:
            if template_path.exists():
                shutil.copy2(template_path, conf_py)
                logger.info(f"📄 Copied conf.py template to {conf_py.relative_to(repo_root)}")
                template_found = True
                break
                
        if not template_found:
            # Create basic conf.py
            with open(conf_py, "w", encoding="utf-8") as f:
                f.write(create_conf_py_content(repo_root.name))
            logger.info(f"📄 Created basic conf.py at {conf_py.relative_to(repo_root)}")
            
    logger.info("✅ Documentation structure setup complete")

def create_index_content(project_name: str) -> str:
    """Create content for index.md file."""
    return f"""# {project_name} Documentation

Welcome to the {project_name} documentation!

## Contents

