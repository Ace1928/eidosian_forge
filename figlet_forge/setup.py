#!/usr/bin/env python3
"""
Figlet Forge setup.py file.
For compatibility with older tools, use this instead of pyproject.toml.
For modern usage, prefer pyproject.toml.
"""

import os
import re

from setuptools import find_packages, setup

# Extract version directly from version.py without importing
version_file = os.path.join("src", "figlet_forge", "version.py")
with open(version_file, encoding="utf-8") as f:
    version_content = f.read()

# Extract metadata using regex patterns
version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', version_content)
author_match = re.search(r'__author__\s*=\s*["\']([^"\']+)["\']', version_content)
author_email_match = re.search(
    r'__author_email__\s*=\s*["\']([^"\']+)["\']', version_content
)
description_match = re.search(
    r'__description__\s*=\s*["\']([^"\']+)["\']', version_content
)

__version__ = version_match.group(1) if version_match else "0.1.0"
__author__ = author_match.group(1) if author_match else "Lloyd Handyside"
__author_email__ = (
    author_email_match.group(1) if author_email_match else "ace1928@gmail.com"
)
__description__ = (
    description_match.group(1)
    if description_match
    else "Enhanced Figlet System with colorized ANSI support and Unicode rendering"
)

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="figlet_forge",
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ace1928/figlet_forge",
    project_urls={
        "Documentation": "https://figlet-forge.readthedocs.io/",
        "Source": "https://github.com/Ace1928/figlet_forge",
        "Tracker": "https://github.com/Ace1928/figlet_forge/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing :: Fonts",
        "Topic :: Artistic Software",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "colorama>=0.4.3; sys_platform == 'win32'",
        "importlib-resources>=1.4.0; python_version < '3.9'",
        "typing-extensions>=4.0.0; python_version < '3.8'",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
            "tox>=4.0.0",
            "twine>=4.0.0",
            "build>=1.0.0",
            "termcolor>=2.0.0",
            "pyfiglet>=0.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "figlet_forge=figlet_forge.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
