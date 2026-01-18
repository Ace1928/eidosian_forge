#!/usr/bin/env python3
# 🌀 Setup with Eidosian thoroughness – bridging all dimensions seamlessly! 🌉
"""
╔══════════════════════════════════════════════════════════════════════════╗
║ 🌀 EIDOSIAN SETUP SCRIPT: WHERE FUNCTION MEETS FORM                     ║
║ 📐 PRECISION ARCHITECTURE FOR PACKAGE DISTRIBUTION                       ║
╚══════════════════════════════════════════════════════════════════════════╝

Legacy setup script for Ollama Forge. 
Each import path represents a layer of resilience 🛡️
Each fallback embodies recursive adaptation 🔄

📚 Eidosian Principles in Action:
1. 🎯 Contextual Integrity - No redundant imports, each path serves a purpose
2. 😄 Humor as Cognitive Leverage - "Epicenter of synergy" lightens complex imports
3. 🗜️ Exhaustive But Concise - Complete error handling without verbosity
4. 🌊 Flow Like a River - Smooth transitions between import attempts
5. 🔧 Hyper-Personal Yet Universal - Works locally or in any CI environment
6. 🔄 Recursive Refinement - Multiple fallback strategies for robustness
7. ✨ Precision as Style - Clean variable assignments with minimal overhead
8. ⚡ Velocity as Intelligence - Fast import paths with optimal checking
9. 🏛️ Structure as Control - Hierarchical import strategy with clear boundaries
10. 🔍 Self-Awareness as Foundation - Import failures trigger intelligent defaults
"""

import setuptools
import os
import sys
import importlib.util
from pathlib import Path
import time
from typing import Callable, cast

# ⏱️ Track import strategy performance
start_time = time.time()

# 🧠 Self-awareness check - validate environment before proceeding
if sys.version_info < (3, 8):
    sys.exit("🚨 Python 3.8+ required - your time machine is stuck in the past! 🕰️")

# Type-safe fallback defaults
package_name_normalized: str = "ollama_forge"
version: str = "0.1.9"
author: str = "Lloyd Handyside, Eidos"
author_email: str = "ace1928@gmail.com, syntheticeidos@gmail.com"
description: str = "Python client library and CLI for Ollama"
import_path: str = "⚠️ Initial defaults"

# 🧩 Import Strategy: Layered resilience with graceful degradation
try:
    # 🥇 Primary path: Direct module access for maximum performance
    config_path = Path(__file__).parent / "ollama_forge" / "config.py"
    if config_path.exists():
        # Import the config module dynamically to avoid import errors
        spec = importlib.util.spec_from_file_location("dynamic_config", config_path)
        if spec is not None and spec.loader is not None:
            dynamic_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dynamic_config)
            
            # Access configurations with proper type annotations
            package_name_normalized = cast(str, getattr(dynamic_config, "PACKAGE_NAME_NORMALIZED", package_name_normalized))
            get_version_string: Callable[[], str] = getattr(dynamic_config, "get_version_string")
            get_author_string: Callable[[], str] = getattr(dynamic_config, "get_author_string")
            get_email_string: Callable[[], str] = getattr(dynamic_config, "get_email_string")
            
            # Get values with proper type casting
            version = get_version_string()
            author = get_author_string()
            author_email = get_email_string()
            import_path = "🔍 Direct config module"
    else:
        # 🥈 Secondary path: Package import with namespace resolution
        sys.path.insert(0, os.path.abspath('.'))
        try:
            ollama_forge_config = importlib.import_module("ollama_forge.config")
            
            # Access configurations with type safety
            package_name_normalized = cast(str, getattr(ollama_forge_config, "PACKAGE_NAME_NORMALIZED", package_name_normalized))
            get_version_string = cast(Callable[[], str], getattr(ollama_forge_config, "get_version_string"))
            get_author_string = cast(Callable[[], str], getattr(ollama_forge_config, "get_author_string"))
            get_email_string = cast(Callable[[], str], getattr(ollama_forge_config, "get_email_string"))
            
            # Get values with proper type casting
            version = get_version_string()
            author = get_author_string()
            author_email = get_email_string()
            import_path = "🔁 Package namespace resolution"
        except (ImportError, AttributeError):
            # Fallback to default values already set
            import_path = "⚠️ Fallback to default values (namespace import failed)"
except Exception as e:
    # 🛡️ Fallback shield: Default values ensure continuity
    import_path = f"⚠️ Fallback values (exception: {type(e).__name__})"

# ⚡ Performance metrics - because velocity matters
import_time = time.time() - start_time

# 📦 Package registration: Minimal yet complete
setuptools.setup(
    name=package_name_normalized,  # 🏷️ Identity
    version=version,               # 🔢 Semantic versioning 
    author=author,                 # 👤 Creator attribution
    author_email=author_email,     # 📫 Contact vector
    description=description,       # 📝 Purpose statement
)

# 🔄 Self-reflection output - no wasted cycles
if os.environ.get("OLLAMA_FORGE_DEBUG") == "1":
    print(f"⚙️ Setup initialized via {import_path}")
    print(f"⏱️ Import strategy completed in {import_time:.6f}s")
    print(f"📦 Package: {package_name_normalized} v{version}")
