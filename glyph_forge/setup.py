#!/usr/bin/env python3
"""Auto-generated setup.py from pyproject.toml.

Do not edit manually. Run: python scripts/sync_setup.py --write
"""
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"

def read_long_description() -> str:
    if README.exists():
        return README.read_text(encoding="utf-8")
    return "Eidosian Glyph Forge - Image to ASCII/ANSI Converter."

setup(
    name="glyph_forge",
    version="0.1.0",
    description="Eidosian Glyph Forge - Image to ASCII/ANSI Converter.",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Lloyd Handyside",
    author_email="ace1928@gmail.com",
    license="MIT",
    python_requires=">=3.12",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=['pillow>=9.0.0', 'numpy>=1.26.0,<2.0', 'rich>=13.0.0', 'typer>=0.9.0', 'pyfiglet>=0.8.0', 'PyYAML>=6.0'],
    extras_require={'dev': ['pytest>=8.0.0', 'pytest-benchmark>=5.2.0', 'pytest-cov>=6.0.0', 'pytest-html>=4.2.0', 'pytest-mock>=3.15.0', 'pytest-profiling>=1.8.0'], 'streaming': ['opencv-python>=4.8.0,<4.13', 'yt-dlp>=2024.12.0', 'mss>=9.0.0'], 'browser': ['playwright>=1.40.0'], 'tui': ['textual>=0.50.0'], 'audio': ['pygame>=2.5.0', 'simpleaudio>=1.0.4'], 'assets': ['requests>=2.31.0'], 'clipboard': ['pyperclip>=1.9.0'], 'all': ['pytest>=8.0.0', 'pytest-benchmark>=5.2.0', 'pytest-cov>=6.0.0', 'pytest-html>=4.2.0', 'pytest-mock>=3.15.0', 'pytest-profiling>=1.8.0', 'opencv-python>=4.8.0,<4.13', 'yt-dlp>=2024.12.0', 'mss>=9.0.0', 'playwright>=1.40.0', 'textual>=0.50.0', 'pygame>=2.5.0', 'simpleaudio>=1.0.4', 'requests>=2.31.0', 'pyperclip>=1.9.0']},
    entry_points={"console_scripts": ['glyph-forge=glyph_forge.cli:main']},
    keywords=['ascii-art', 'ansi', 'image-processing', 'pillow'],
    classifiers=['Development Status :: 4 - Beta', 'Intended Audience :: Developers', 'License :: OSI Approved :: MIT License', 'Programming Language :: Python :: 3.12'],
)
