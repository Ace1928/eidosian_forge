"""
Template content constants for Eidosian repository generation.

This module centralizes all template strings, configuration content,
and boilerplate text used across the repository generator system.
"""
from typing import Dict, Any, List
import textwrap
from datetime import datetime

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ README Templates                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

README_TEMPLATE = """# 🔮 $repo_name

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Updated](https://img.shields.io/badge/updated-$current_date-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**Universal Eidosian Monorepo Structure**

## Features 🚀

- 📦 Universal project structure optimized for any language
- 🔄 Cross-language interoperability with unified interfaces
- 📚 Comprehensive documentation system
- ⚙️ Streamlined build and testing pipeline
- 🧩 Modular architecture for ultimate composability
- 🔍 Integrated quality assurance workflows

## Structure 🏗️

```
.
├── projects/         # Language-specific projects
├── libs/             # Shared libraries and components
├── tools/            # Development and build tools
├── scripts/          # Automation scripts
├── docs/             # Documentation
├── tests/            # Integrated test suite
├── benchmarks/       # Performance benchmarks
├── examples/         # Example code and tutorials
└── ci/               # Continuous integration configuration
```

## Getting Started 🏁

Clone this repository and explore the structure to get familiar with the organization.

## Contributing 👥

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md).

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Created with 💜 using Eidosian Repo Forge.
"""

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Configuration Files                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

GITIGNORE_CONTENT = """# 🔥 Eidosian-optimized .gitignore 🔥

# ╔══ Languages ══════════════════════════╗

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.coverage
htmlcov/

# Node.js
node_modules/
npm-debug.log
yarn-error.log
yarn-debug.log
.npm/
.yarn/
.pnp.*
package-lock.json
.node_repl_history
.env
.env.test

# Go
*.o
*.a
*.so
_obj
_test
*.exe
*.test
*.prof
*.out
.glide/
vendor/

# Rust
target/
**/*.rs.bk
Cargo.lock

# ╔══ Environments ═══════════════════════╗
.env
.venv
env/
venv/
ENV/
.virtualenv/

# ╔══ Editors ═════════════════════════════╗
# VSCode
.vscode/*
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
!.vscode/extensions.json
.history

# JetBrains
.idea/
*.iml
*.iws
*.ipr
*.iws
.idea_modules/
out/

# ╔══ OS Files ════════════════════════════╗
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# ╔══ Logs & Databases ═══════════════════╗
*.log
*.sql
*.sqlite
*.db

# ╔══ Temp Files ═══════════════════════════╗
*~
*.swp
*.swo
*.tmp
*.bak
"""

EDITORCONFIG_CONTENT = """# 🎯 Eidosian EditorConfig - Universal Code Harmony

# Top-most EditorConfig file
root = true

# Universal defaults for all files
[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 2
max_line_length = 100

# Python files
[*.py]
indent_size = 4
max_line_length = 88

# Go files
[*.go]
indent_style = tab
indent_size = 4

# Rust files
[*.rs]
indent_size = 4
max_line_length = 100

# JavaScript/TypeScript files
[*.{js,jsx,ts,tsx}]
indent_size = 2
max_line_length = 80

# Java files
[*.java]
indent_size = 4
max_line_length = 120

# C/C++ files
[*.{c,cpp,h,hpp}]
indent_size = 4
max_line_length = 100

# Shell scripts
[*.{sh,bash,zsh}]
indent_size = 2

# Markdown files
[*.md]
trim_trailing_whitespace = false
max_line_length = off

# YAML files
[*.{yaml,yml}]
indent_size = 2

# JSON files
[*.json]
indent_size = 2
insert_final_newline = false

# XML files
[*.xml]
indent_size = 2

# Makefiles require tabs
[Makefile]
indent_style = tab

# Documentation files
[*.{txt,rst}]
max_line_length = 79

# CI configuration files
[*.{toml,ini}]
indent_size = 2
"""

EIDOSIAN_CONFIG_TEMPLATE = """# 🔮 Eidosian Configuration

# Repository metadata
repo_name: "$repo_name"
created_at: "$current_date"
version: "0.1.0"

# Structure configuration
structure:
  enforce_consistent_structure: true
  allow_custom_directories: true
  require_documentation: true
  enforce_testing: true

# Code style
code_style:
  python:
    formatter: "black"
    line_length: 88
    docstring_style: "sphinx"
  javascript:
    formatter: "prettier"
    line_length: 80
  go:
    formatter: "gofmt"
  rust:
    formatter: "rustfmt"

# Documentation
documentation:
  require_readme: true
  require_contributing: true
  require_code_of_conduct: true
  require_license: true
  api_docs_generator: "sphinx"

# CI/CD
ci_cd:
  required_checks:
    - "lint"
    - "test"
    - "build"
    - "docs"
  platforms:
    - "github-actions"
    - "gitlab-ci"

# Dependencies
dependencies:
  allow_direct_dependencies: true
  require_dependency_lockfiles: true
  security_scanning: true

# Security
security:
  enforce_security_policy: true
  require_vulnerability_reporting: true
  security_scanning_on_pr: true

# Community
community:
  require_code_of_conduct: true
  issue_templates: true
  pr_templates: true
"""

CI_CONFIG_TEMPLATE = """# 🚀 Eidosian CI Pipeline
# Universal CI configuration that adapts to multiple CI systems

name: Eidosian Universal CI

on:
  push:
    branches: [ main, master, develop ]
  pull_request:
    branches: [ main, master, develop ]

jobs:
  lint:
    name: 🧹 Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
          pip install black isort flake8 mypy
      - name: Check code formatting
        run: |
          black . --check
          isort . --check
      - name: Lint with flake8
        run: flake8 .
      - name: Type check with mypy
        run: mypy .

  test:
    name: 🧪 Test
    needs: lint
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.12']
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Run tests
        run: |
          pytest --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: false

  docs:
    name: 📚 Docs
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[docs]"
      - name: Build docs
        run: |
          cd docs
          make html
      - name: Upload docs
        uses: actions/upload-artifact@v3
        with:
          name: docs
          path: docs/_build/html

  build:
    name: 📦 Build
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      - name: Build package
        run: python -m build
      - name: Check package
        run: twine check dist/*
"""

CONTRIBUTING_TEMPLATE = """# 🤝 Contributing to $repo_name

Thank you for considering contributing to this project! This document outlines the process and guidelines for contributing.

## 🌟 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🔄 Development Workflow

1. **Fork the repository** and clone it locally
2. **Create a new branch** for your feature or bug fix
3. **Make your changes** following our coding conventions
4. **Write or update tests** to verify your changes
5. **Run the test suite** to ensure nothing breaks
6. **Update documentation** if necessary
7. **Submit a pull request** with a clear description of the changes

## 🧪 Testing

Run the test suite with:

```bash
pytest
```

## 📝 Coding Standards

- Follow the existing code style and formatting
- Write comprehensive docstrings for all functions, classes, and modules
- Include type hints for all function parameters and return values
- Keep functions small and focused on a single task
- Write meaningful variable and function names

## 📚 Documentation

Update documentation to reflect any changes:

- README.md for user-facing changes
- Code docstrings for API changes
- Wiki pages for conceptual documentation

## 🔍 Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the CHANGELOG.md with a description of the changes
3. The PR will be merged once it receives approval from maintainers

## 🐛 Reporting Bugs

Please report bugs by opening an issue that includes:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment information (OS, Python version, etc.)

## 💡 Feature Requests

Feature requests are welcome! Please submit an issue that includes:

- A clear, descriptive title
- A detailed description of the proposed feature
- Any relevant examples or use cases

Thank you for contributing! 🎉
"""

CODE_OF_CONDUCT_CONTENT = """# 📜 Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socio-economic status,
nationality, personal appearance, race, religion, or sexual identity
and orientation.

## Our Standards

Examples of behavior that contributes to creating a positive environment include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior include:

* The use of sexualized language or imagery and unwelcome sexual attention or advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information without explicit permission
* Other conduct which could reasonably be considered inappropriate in a professional setting

## Enforcement

Violations of the Code of Conduct may be reported to the project team. All
complaints will be reviewed and investigated promptly and fairly. The project team is
obligated to maintain confidentiality with regard to the reporter of an incident.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org),
version 2.0, available at
https://www.contributor-covenant.org/version/2/0/code_of_conduct.html.
"""

LICENSE_TEMPLATE = """MIT License

Copyright (c) $current_year

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

SECURITY_TEMPLATE = """# 🔐 Security Policy

## Supported Versions

Currently supported versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

To report a security issue, please email $author_email with a description of the issue,
the steps you took to create the issue, affected versions, and if known, mitigations for the issue.

We aim to respond within 48 hours, and will work with you to understand and address the issue promptly.

## Security Measures

This project implements the following security measures:

- Regular dependency updates
- Code scanning for vulnerabilities
- Security-focused code reviews
- Automated testing for security issues

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm receipt of the vulnerability report
2. Investigate and determine impact
3. Develop and test a fix
4. Release a security update
5. Publicly disclose the issue after the fix is released
"""

CHANGELOG_TEMPLATE = """# 📝 Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🚀 Added
- Initial repository structure
- Core framework components
- Basic documentation

### 🔧 Changed
- N/A

### 🐛 Fixed
- N/A

### 🔒 Security
- N/A

## [0.1.0] - $current_date

### 🚀 Added
- Initial release with Eidosian Repo Forge
"""

# Language-specific project templates
PROJECT_READMES = {
    "python": """# 🐍 $project_name

Python project within the Eidosian monorepo structure.

## Installation

```bash
pip install -e .
```

## Usage

```python
from $project_name import main

main.run()
```
""",
    "nodejs": """# 🟢 $project_name

Node.js project within the Eidosian monorepo structure.

## Installation

```bash
npm install
```

## Usage

```javascript
const main = require('./src/main');

main.run();
```
""",
    "go": """# 🔵 $project_name

Go project within the Eidosian monorepo structure.

## Building

```bash
go build ./...
```

## Usage

```go
package main

import (
	"$project_name/pkg"
)

func main() {
	pkg.Run()
}
```
""",
    "rust": """# 🦀 $project_name

Rust project within the Eidosian monorepo structure.

## Building

```bash
cargo build
```

## Usage

```rust
use $project_name::run;

fn main() {
    run();
}
```
""",
    "default": """# 📦 $project_name

Project within the Eidosian monorepo structure.

## Getting Started

See the documentation for details on how to use this project.
"""
}

# Configuration files for specific languages
PROJECT_CONFIG_FILES = {
    "python": {
        "pyproject.toml": """[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "python_project"
version = "0.1.0"
description = "Python project within Eidosian monorepo"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Eidosian Team", email = "team@example.com"},
]
dependencies = [
    "requests>=2.28.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]

[tool.black]
line-length = 88
target-version = ["py310"]

[tool.mypy]
python_version = "3.10"
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
"""
    },
    "nodejs": {
        "package.json": """{
  "name": "nodejs_project",
  "version": "0.1.0",
  "description": "Node.js project within Eidosian monorepo",
  "main": "src/index.js",
  "scripts": {
    "test": "jest",
    "lint": "eslint .",
    "start": "node src/index.js"
  },
  "author": "Eidosian Team <team@example.com>",
  "license": "MIT",
  "dependencies": {
    "axios": "^1.6.0"
  },
  "devDependencies": {
    "eslint": "^8.54.0",
    "jest": "^29.7.0"
  }
}
"""
    },
    "go": {
        "go.mod": """module go_project

go 1.21

require (
	github.com/stretchr/testify v1.8.4
)
"""
    },
    "rust": {
        "Cargo.toml": """[package]
name = "rust_project"
version = "0.1.0"
edition = "2021"
authors = ["Eidosian Team <team@example.com>"]
license = "MIT"
description = "Rust project within Eidosian monorepo"

[dependencies]
tokio = { version = "1.34", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }

[dev-dependencies]
tokio-test = "0.4"
"""
    }
}

# Main files for each language
PROJECT_MAIN_FILES = {
    "python": {
        "__init__.py": """\"\"\"
Python project module.

This module provides the core functionality for this project.
\"\"\"

__version__ = "0.1.0"
""",
        "main.py": """\"\"\"
Main entry point for the Python project.
\"\"\"
from typing import Dict, Any


def run() -> Dict[str, Any]:
    \"\"\"
    Run the main functionality of the project.
    
    Returns:
        Dictionary with the results
    \"\"\"
    return {
        "status": "success",
        "message": "Hello from Python project!"
    }


if __name__ == "__main__":
    result = run()
    print(f"Result: {result}")
"""
    },
    "nodejs": {
        "index.js": """/**
 * Main entry point for the Node.js project.
 * @module index
 */

'use strict';

/**
 * Run the main functionality of the project.
 * @returns {Object} Result object
 */
function run() {
  return {
    status: 'success',
    message: 'Hello from Node.js project!'
  };
}

// Export the run function
module.exports = {
  run
};

// Run if called directly
if (require.main === module) {
  const result = run();
  console.log(`Result: ${JSON.stringify(result)}`);
}
"""
    },
    "go": {
        "main.go": """// Main package for the Go project
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
)

// Result represents the output of the run function
type Result struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

// Run executes the main functionality of the project
func Run() Result {
	return Result{
		Status:  "success",
		Message: "Hello from Go project!",
	}
}

func main() {
	result := Run()
	
	// Convert result to JSON
	jsonData, err := json.Marshal(result)
	if err != nil {
		log.Fatalf("Error marshalling result: %v", err)
	}
	
	fmt.Printf("Result: %s\\n", string(jsonData))
	os.Exit(0)
}
"""
    },
    "rust": {
        "main.rs": """//! Main entry point for the Rust project.

use serde::{Deserialize, Serialize};

/// Result of the run function
#[derive(Debug, Serialize, Deserialize)]
pub struct RunResult {
    status: String,
    message: String,
}

/// Run the main functionality of the project
pub fn run() -> RunResult {
    RunResult {
        status: String::from("success"),
        message: String::from("Hello from Rust project!"),
    }
}

fn main() {
    let result = run();
    println!("Result: {:?}", result);
}
"""
    }
}
