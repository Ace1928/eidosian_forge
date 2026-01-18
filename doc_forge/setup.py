#!/usr/bin/env python3
# 🌀 Eidosian Documentation System - Package Setup
"""
Doc Forge - Package Setup

This script configures the Doc Forge package for distribution.
It is designed to be compatible with both setuptools and pip install.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read version from VERSION file if it exists, otherwise use default
version_file = Path(__file__).parent / "VERSION"
if version_file.exists():
    with open(version_file, "r", encoding="utf-8") as f:
        version = f.read().strip()
else:
    version = "0.1.0"  # Default version

# Read long description from README.md if it exists
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Universal Documentation Management System with Eidosian principles"

# Core dependencies
install_requires = [
    "sphinx>=8.0.0",
    "sphinx-rtd-theme>=1.3.0",
    "sphinx-autoapi>=3.0.0",
    "myst-parser>=2.0.0",
    "sphinx-copybutton>=0.5.2",
    "sphinx-autodoc-typehints>=1.25.0",
    "sphinxcontrib-mermaid>=0.9.2",
    "sphinx-sitemap>=2.5.0",
    "sphinx-tabs>=3.4.1",
    "sphinx-markdown-tables>=0.0.17",
    "sphinx-notfound-page>=0.8.3",
    "sphinx-inline-tabs>=2023.4.21",
    "sphinxext-opengraph>=0.9.1",
    "sphinx-design>=0.5.0",
    "pyyaml>=6.0.1",
    "colorama>=0.4.6",
    "typer>=0.9.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=23.0.0",
        "isort>=5.10.0",
        "mypy>=1.0.0",
        "flake8>=6.0.0",
    ]
}

if __name__ == "__main__":
    setup(
        name="doc_forge",
        version=version,
        description="Universal Documentation Management System with Eidosian principles",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Lloyd Handyside",
        author_email="ace1928@gmail.com",
        maintainer="Lloyd Handyside",
        maintainer_email="ace1928@gmail.com",
        url="https://doc-forge.readthedocs.io/",
        project_urls={
            "Source": "https://github.com/Ace1928/doc_forge",
            "Documentation": "https://doc-forge.readthedocs.io/",
            "Issue Tracker": "https://github.com/Ace1928/doc_forge/issues",
        },
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Software Development :: Documentation",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
        python_requires=">=3.8",
        install_requires=install_requires,
        extras_require=extras_require,
        entry_points={
            "console_scripts": [
                "doc-forge=doc_forge:main",
            ],
        },
        include_package_data=True,
        zip_safe=False,
    )
