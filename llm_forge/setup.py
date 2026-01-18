#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Forge: Setup Configuration

This file exists primarily for backward compatibility with tools that
don't fully support PEP 517/518 standards. The authoritative configuration
is maintained in pyproject.toml.

Author: Lloyd Handyside <ace1928@gmail.com>
License: MIT
Version: 0.1.0
"""

from __future__ import annotations

import os
import sys
from typing import List

# Ensure proper Python version
if sys.version_info < (3, 8):
    raise RuntimeError(
        "LLM Forge requires Python 3.8+ (current: {}.{}.{})".format(
            *sys.version_info[:3]
        )
    )

try:
    from setuptools import find_packages, setup
except ImportError:
    print("Error: setuptools is required. Please install it first.")
    sys.exit(1)


def read_file(filename: str) -> str:
    """
    Read file content with error handling.

    Args:
        filename: Path to file relative to setup.py

    Returns:
        Content of the file as string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    with open(os.path.join(os.path.dirname(__file__), filename), encoding="utf-8") as f:
        return f.read()


def parse_requirements(filename: str) -> List[str]:
    """
    Parse requirements file into a list of package requirements.

    Args:
        filename: Path to requirements file

    Returns:
        List of parsed requirements
    """
    try:
        return [
            line.strip()
            for line in read_file(filename).splitlines()
            if line.strip() and not line.startswith("#")
        ]
    except FileNotFoundError:
        return []


# Core project metadata
PROJECT_NAME = "llm_forge"
VERSION = "0.1.0"
DESCRIPTION = (
    "A recursive framework for language model development, "
    "orchestration, and deployment"
)
AUTHOR = "Lloyd Handyside"
AUTHOR_EMAIL = "ace1928@gmail.com"
PROJECT_URL = "https://github.com/Ace1928/llm_forge"
PYTHON_REQUIRES = ">=3.8, <3.13"

# Dependency management
INSTALL_REQUIRES = parse_requirements("requirements.txt")
DEV_REQUIRES = parse_requirements("requirements-dev.txt")

# Match extras_require with pyproject.toml's structure
DEV_EXTRAS = [
    "pytest>=7.0.0,<8.0.0",
    "pytest-cov>=4.0.0,<5.0.0",
    "mypy>=1.0.0,<2.0.0",
    "black>=23.0.0,<24.0.0",
    "isort>=5.12.0,<6.0.0",
    "ruff>=0.0.275,<0.1.0",
    "pre-commit>=3.0.0,<4.0.0",
]

DOCS_EXTRAS = [
    "sphinx>=6.0.0,<7.0.0",
    "sphinx-rtd-theme>=1.2.0,<2.0.0",
    "sphinx-autodoc-typehints>=1.22.0,<2.0.0",
    "myst-parser>=2.0.0,<3.0.0",
]

TESTING_EXTRAS = [
    "pytest>=7.0.0,<8.0.0",
    "pytest-cov>=4.0.0,<5.0.0",
    "tox>=4.0.0,<5.0.0",
]

if __name__ == "__main__":
    setup(
        # Core metadata (duplicated from pyproject.toml for compatibility)
        name=PROJECT_NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=read_file("README.md"),
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=PROJECT_URL,
        # Package discovery
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        # Dependencies
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        extras_require={
            "dev": DEV_EXTRAS,
            "docs": DOCS_EXTRAS,
            "testing": TESTING_EXTRAS,
            "all": ["llm_forge[dev,docs,testing]"],
        },
        # Entry points
        entry_points={
            "console_scripts": [
                "llm-forge=llm_forge.cli.main:main",
            ],
        },
        # Classifiers
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Libraries",
        ],
        # Additional metadata
        license="MIT",
        keywords=[
            "llm",
            "forge",
            "language model",
            "processing",
            "orchestration",
            "ai",
            "generative",
            "pipeline",
        ],
        project_urls={
            "Homepage": "https://github.com/Ace1928/llm_forge",
            "Repository": "https://github.com/Ace1928/llm_forge",
            "Documentation": "https://github.com/Ace1928/llm_forge/docs",
            "Bug Tracker": "https://github.com/Ace1928/llm_forge/issues",
            "Change Log": "https://github.com/Ace1928/llm_forge/blob/main/CHANGELOG.md",
        },
        # Include package data
        include_package_data=True,
        zip_safe=False,
    )
