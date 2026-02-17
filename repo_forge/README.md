# 📐 Repo Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Blueprint of Eidos.**

> _"Standards are the scaffolding of emergence."_

## 📐 Overview

`repo_forge` is the custodian of the Eidosian ecosystem's structure. It ensures that every Forge adheres to the strict organizational protocols required for seamless integration, automated testing, and recursive documentation.

## 🏗️ Architecture

- **Project Generator (`generators/`)**: Standardized templates for new Forges, scripts, and libraries.
- **Diagnostics (`core/diagnostics.py`)**: Checks repository health, identifying missing files (`README.md`, `pyproject.toml`) or broken dependency links.
- **Registry (`core/registry.py`)**: Centralized manifest of all active Forges and their metadata.
- **Directory Atlas**: Generates linked maps of the filesystem.

## 🔗 System Integration

- **Documentation Forge**: `doc_forge` uses the registry to find directories to scan.
- **Eidos MCP**: Provides tools for system-wide info (`system_info`) and repo analysis.

## 🚀 Usage

### Creating a New Forge

```bash
python -m repo_forge.cli create my_new_forge
```

### Running Repo Health Check

```bash
python -m repo_forge.cli doctor
```
