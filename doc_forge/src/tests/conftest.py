#!/usr/bin/env python3
# 🌀 Pytest Configuration for Doc Forge Tests
"""
Pytest Configuration - Universal Test Setup with Eidosian Precision

This module provides fixtures and configuration for pytest,
following Eidosian principles of structure, consistency, and clarity.
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Set, Any, Generator, Callable

# Add the project root to the Python path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Constants for test paths
REPO_ROOT = repo_root
SRC_DIR = repo_root / "src"
TESTS_DIR = repo_root / "tests"
TEST_DATA_DIR = TESTS_DIR / "data"

# Create test data directory if it doesn't exist
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

@pytest.fixture
def repo_root_path() -> Path:
    """Fixture that provides the repository root path."""
    return REPO_ROOT

@pytest.fixture
def src_dir_path() -> Path:
    """Fixture that provides the source directory path."""
    return SRC_DIR

@pytest.fixture
def tests_dir_path() -> Path:
    """Fixture that provides the tests directory path."""
    return TESTS_DIR

@pytest.fixture
def test_data_dir() -> Path:
    """Fixture that provides the test data directory path."""
    return TEST_DATA_DIR

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Fixture that provides a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def temp_docs_dir(temp_dir) -> Generator[Path, None, None]:
    """Fixture that provides a temporary docs directory structure for tests."""
    docs_dir = temp_dir / "docs"
    
    # Create basic docs directory structure
    (docs_dir / "_static").mkdir(parents=True, exist_ok=True)
    (docs_dir / "_templates").mkdir(parents=True, exist_ok=True)
    (docs_dir / "user_docs" / "getting_started").mkdir(parents=True, exist_ok=True)
    (docs_dir / "user_docs" / "guides").mkdir(parents=True, exist_ok=True)
    (docs_dir / "user_docs" / "reference").mkdir(parents=True, exist_ok=True)
    (docs_dir / "auto_docs" / "api").mkdir(parents=True, exist_ok=True)
    (docs_dir / "ai_docs").mkdir(parents=True, exist_ok=True)
    
    # Create a basic index.md file
    with open(docs_dir / "index.md", "w") as f:
        f.write("# Test Documentation\n\nThis is a test documentation index.\n")
    
    # Create a sample user doc
    with open(docs_dir / "user_docs" / "getting_started" / "index.md", "w") as f:
        f.write("# Getting Started\n\nThis is the getting started guide.\n")
    
    # Create a sample auto doc
    with open(docs_dir / "auto_docs" / "api" / "index.md", "w") as f:
        f.write("# API Documentation\n\nThis is the API documentation.\n")
    
    yield docs_dir

@pytest.fixture
def sample_toc_structure() -> Dict[str, Any]:
    """Fixture that provides a sample TOC structure for tests."""
    return {
        "getting_started": {
            "title": "Getting Started",
            "items": [
                {
                    "title": "Installation",
                    "url": "user_docs/getting_started/installation.html",
                    "priority": 10
                },
                {
                    "title": "Quickstart",
                    "url": "user_docs/getting_started/quickstart.html",
                    "priority": 20
                }
            ]
        },
        "user_guide": {
            "title": "User Guide",
            "items": [
                {
                    "title": "Basic Usage",
                    "url": "user_docs/guides/basic_usage.html",
                    "priority": 30
                },
                {
                    "title": "Advanced Features",
                    "url": "user_docs/guides/advanced_features.html",
                    "priority": 40
                }
            ]
        },
        "reference": {
            "title": "API Reference",
            "items": [
                {
                    "title": "Module Reference",
                    "url": "auto_docs/api/module_reference.html",
                    "priority": 50
                }
            ]
        }
    }

@pytest.fixture
def mock_document_metadata(temp_docs_dir) -> Callable:
    """
    Fixture that provides a function to create DocumentMetadata instances.
    
    Returns:
        Function that creates DocumentMetadata instances
    """
    from doc_forge.source_discovery import DocumentMetadata
    
    def create_metadata(
        filename: str, 
        title: str = "", 
        category: str = "user", 
        section: str = "getting_started", 
        priority: int = 50
    ) -> DocumentMetadata:
        # Determine the path based on category and section
        if category == "user":
            path = temp_docs_dir / "user_docs" / section / filename
        elif category == "auto":
            path = temp_docs_dir / "auto_docs" / section / filename
        elif category == "ai":
            path = temp_docs_dir / "ai_docs" / section / filename
        else:
            path = temp_docs_dir / filename
            
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a simple file if it doesn't exist
        if not path.exists():
            with open(path, "w") as f:
                f.write(f"# {title or path.stem.title()}\n\nSample content.\n")
        
        return DocumentMetadata(
            path=path,
            title=title,
            category=category,
            section=section,
            priority=priority
        )
    
    return create_metadata

@pytest.fixture
def mock_toc_tree_manager(temp_docs_dir) -> Any:
    """
    Fixture that provides a TocTreeManager instance for testing.
    
    Returns:
        TocTreeManager instance
    """
    from doc_forge.update_toctrees import TocTreeManager
    return TocTreeManager(temp_docs_dir)

@pytest.fixture
def mock_documentation_discovery(temp_docs_dir) -> Any:
    """
    Fixture that provides a DocumentationDiscovery instance for testing.
    
    Returns:
        DocumentationDiscovery instance
    """
    from doc_forge.source_discovery import DocumentationDiscovery
    return DocumentationDiscovery(docs_dir=temp_docs_dir)

# Import markers to make them available
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
