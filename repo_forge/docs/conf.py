#!/usr/bin/env python3
# ╔═══════════════════════════════════════════════════════════╗
# ║  E I D O S I A N   D O C U M E N T A T I O N   S Y S T E M  ║
# ╚═══════════════════════════════════════════════════════════╝
"""
Sphinx configuration for Repo Forge documentation.

This module sets up the Sphinx documentation generator to produce
elegant, comprehensive documentation that adheres perfectly to
Eidosian principles of clarity, structure, and recursive refinement.
"""
import os
import sys
import json
import shutil
import logging
import importlib.util
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 Core Project Identity - Unified Information Architecture
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
project = "Repo Forge"
copyright = f"{datetime.now().year}, Lloyd Handyside, Eidos"
author = "Lloyd Handyside and Eidos"

# Extract version from __init__.py to ensure single source of truth
try:
    import repo_forge
    version = repo_forge.__version__  # 🔢 Semantic versioning - precision matters
    release = version  # 🎯 Release tracking - keep in sync for mathematical truth
except (ImportError, AttributeError):
    version = "0.1.0"  # Default if extraction fails
    release = version

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 Comprehensive Logging - Self-Awareness Foundation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
)
logger = logging.getLogger("eidosian_docs")  # 🧠 Centralized logger for all operations

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🏛️ Universal Architecture - Structure as Control
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"  # 📁 Documentation home
BUILD_DIR = DOCS_DIR / "_build"  # 🏗️ Where artifacts materialize
SOURCE_DIR = DOCS_DIR / "source"  # 📦 Source documentation
MANUAL_DIR = DOCS_DIR / "manual"  # 📘 Manual documentation
AUTO_DIR = DOCS_DIR / "auto"  # 🤖 Auto-generated documentation
ASSETS_DIR = DOCS_DIR / "assets"  # 🖼️ Documentation assets
sys.path.insert(0, str(REPO_ROOT))  # Path integration for seamless imports

# Create required directories based on universal_doc_structure.md
for directory in [
    DOCS_DIR / "_static",
    DOCS_DIR / "_templates", 
    # Manual docs structure
    MANUAL_DIR,
    *[MANUAL_DIR / lang / category for lang in ["python"] for category in [
        "guides", "api", "design", "examples", "best_practices", 
        "troubleshooting", "security", "changelog", "contributing", "faq"
    ]],
    # Auto-generated docs structure
    AUTO_DIR, 
    AUTO_DIR / "api",
    AUTO_DIR / "models",
    AUTO_DIR / "functions",
    AUTO_DIR / "error_handling",
    AUTO_DIR / "benchmarks",
    AUTO_DIR / "internal",
    AUTO_DIR / "schemas",
    AUTO_DIR / "configuration",
    # Assets structure
    ASSETS_DIR,
    ASSETS_DIR / "images",
    ASSETS_DIR / "diagrams",
    ASSETS_DIR / "css",
    ASSETS_DIR / "fonts",
    # Source structure - Based on Sphinx standards
    SOURCE_DIR,
    SOURCE_DIR / "getting_started",
    SOURCE_DIR / "concepts",
    SOURCE_DIR / "examples",
    SOURCE_DIR / "guides",
    SOURCE_DIR / "reference",
]:
    directory.mkdir(exist_ok=True, parents=True)

# Generate .gitkeep files for empty directories
for dirpath, dirnames, filenames in os.walk(str(DOCS_DIR)):
    if not filenames and not dirnames:
        Path(dirpath, ".gitkeep").touch(exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🗺️ Navigation Taxonomy - Universal Wayfinding System
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENTATION_CATEGORIES = {
    "getting_started": {
        "caption": "📚 Getting Started",
        "description": "Introduction and initial setup",
        "priority": 100,
        "path": SOURCE_DIR / "getting_started",
        "items": []
    },
    "concepts": {
        "caption": "🧩 Core Concepts",
        "description": "Universal repository structure patterns",
        "priority": 90,
        "path": SOURCE_DIR / "concepts",
        "items": []
    },
    "examples": {
        "caption": "📋 Examples",
        "description": "Example repositories and use cases",
        "priority": 80,
        "path": SOURCE_DIR / "examples",
        "items": []
    },
    "guides": {
        "caption": "🧠 Guides & Tutorials",
        "description": "Step-by-step guidance for repository creation",
        "priority": 70,
        "path": SOURCE_DIR / "guides",
        "items": []
    },
    "reference": {
        "caption": "🔍 API Reference",
        "description": "Generator modules and configuration options",
        "priority": 60,
        "path": SOURCE_DIR / "reference",
        "items": []
    },
    "development": {
        "caption": "🛠️ Development",
        "description": "Contributing to Repo Forge",
        "priority": 50,
        "path": SOURCE_DIR / "development",
        "items": []
    },
}

# Generate index files that embody Eidosian clarity
for category_name, category_info in DOCUMENTATION_CATEGORIES.items():
    index_file = category_info["path"] / "index.md"
    if not index_file.exists():
        emoji = category_info["caption"].split(" ")[0]
        title = " ".join(category_info["caption"].split(" ")[1:])
        with open(index_file, "w") as f:
            f.write(f"""# {emoji} {title}

{category_info['description']}

This section provides comprehensive documentation about {category_name.replace('_', ' ')}.
Each subsection is designed with Eidosian principles of clarity and precision.
""")

# Sphinx Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode", 
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",  # For Markdown support
    "sphinx.ext.intersphinx",
]

# Try to import autoapi extension
try:
    import autoapi
    extensions.append("autoapi.extension")
    autoapi_type = "python"
    autoapi_dirs = [str(REPO_ROOT / "repo_forge")]
    autoapi_output_dir = str(AUTO_DIR / "api")
    autoapi_template_dir = str(DOCS_DIR / "_templates" / "autoapi") if (DOCS_DIR / "_templates" / "autoapi").exists() else None
    autoapi_options = [
        'members',
        'undoc-members',
        'private-members',
        'show-inheritance',
        'show-module-summary',
        'special-members',
        'imported-members',
    ]
    autoapi_add_toctree_entry = True
    autoapi_keep_files = True
    autoapi_python_class_content = 'both'
    logger.info(f"🔄 Using autoapi version: {autoapi.__version__}")
except ImportError:
    logger.warning("⚠️ sphinx-autoapi not installed, API documentation will be limited")

# Intersphinx configuration for cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

# Markdown configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 4

# HTML Theme Configuration
html_theme = "furo"
html_static_path = ["_static"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".txt": "markdown",
}

# Theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "announcement": "🔮 Repo Forge: Universal monorepo structure generator",
    "light_css_variables": {
        "color-brand-primary": "#5D44F3",
        "color-brand-content": "#5D44F3",
    },
    "dark_css_variables": {
        "color-brand-primary": "#9285F7",
        "color-brand-content": "#9285F7",
    },
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Ace1928/repo_forge",
            "html": "",
            "class": "fa fa-github fa-2x",
        },
    ],
}

# Ensure custom CSS file exists
custom_css_path = DOCS_DIR / "_static" / "custom.css"
if not custom_css_path.exists():
    with open(custom_css_path, "w") as f:
        f.write("""/* Eidosian Custom Styles */
.eidosian-highlight {
    background-color: rgba(93, 68, 243, 0.1);
    border-left: 3px solid #5D44F3;
    padding: 0.5em 1em;
    margin: 1em 0;
}
div.highlight pre {
    border-radius: 0.3em;
    padding: 1em;
}
table.docutils {
    width: 100%;
    border-collapse: collapse;
}
table.docutils th, table.docutils td {
    padding: 0.5em;
    border: 1px solid #e1e4e5;
}
table.docutils th {
    background-color: #f3f4f7;
}
.principle {
    margin: 1.5em 0;
    padding: 1em;
    border-left: 3px solid #5D44F3;
    background-color: rgba(93, 68, 243, 0.05);
}
.principle-title {
    font-weight: bold;
    color: #5D44F3;
    margin-bottom: 0.5em;
}
""")
    logger.info("📝 Created custom CSS file")

# Create structure index if it doesn't exist
index_path = DOCS_DIR / "index.md"
if not index_path.exists():
    with open(index_path, "w") as f:
        f.write("""# Repo Forge Documentation

Welcome to the comprehensive documentation for Repo Forge, the universal monorepo structure generator.

## 🚀 Getting Started

Jump right in with our [Quick Start Guide](source/getting_started/index.md) to create your first repository structure.

## 🧩 Core Concepts

Understand the [fundamental principles](source/concepts/index.md) behind Eidosian repository structures.

## 📋 Examples & Use Cases

See [real-world examples](source/examples/index.md) of repositories created with Repo Forge.

## 📚 Guides & Tutorials

Follow our [detailed guides](source/guides/index.md) for advanced repository configuration and customization.

## 🔍 API Reference

Explore the [complete API](source/reference/index.md) for advanced programmatic usage.
""")

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

# Configuration to avoid warning spam
nitpicky = True
nitpick_ignore = [
    ("py:class", "Path"),
    ("py:class", "argparse.ArgumentParser"),
    ("py:class", "argparse.Namespace"),
    ("py:data", "typing.Any"),
    ("py:data", "typing.Optional"),
    ("py:data", "typing.Union"),
    ("py:data", "typing.List"),
    ("py:data", "typing.Dict"),
    ("py:data", "typing.Tuple"),
    ("py:data", "typing.Callable"),
]

suppress_warnings = [
    "toc.not_included",
    "autosectionlabel.*",
    "duplicate.label",
    "myst.header",
]

# Exclude patterns for docs processing
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/__pycache__/**"]
templates_path = ["_templates"]
todo_include_todos = True
copybutton_prompt_text = ">>> "

# Function to set up
def setup(app):
    app.add_css_file("custom.css")
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
