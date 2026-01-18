# Contributing to Figlet Forge

> _"Form follows function; elegance emerges from precision."_

Thank you for your interest in contributing to Figlet Forge! This document provides guidelines and workflows for contributing to this project in alignment with Eidosian principles.

## Table of Contents

```ascii
╭──────────────────────────────────────────────────────╮
│ ◈ Eidosian Development Philosophy                    │
│ ◈ Code of Conduct                                    │
│ ◈ Development Environment Setup                      │
│ ◈ Pull Request Process                               │
│ ◈ Style Guidelines                                   │
│ ◈ Testing Requirements                               │
│ ◈ Documentation Standards                            │
│ ◈ Version Control Practices                          │
│ ◈ Release Process                                    │
╰──────────────────────────────────────────────────────╯
```

## Eidosian Development Philosophy

Figlet Forge follows Eidosian principles which emphasize:

1. **Contextual Integrity** — Code elements justify their existence or face removal
2. **Exhaustive But Concise** — Complete coverage with minimal verbosity
3. **Flow Like Water** — Operations chain seamlessly with type-guided momentum
4. **Recursive Refinement** — Implementation evolves through iterative self-improvement
5. **Precision as Style** — Beauty through perfect alignment with purpose

These principles guide all development decisions and code quality standards.

## Code of Conduct

This project adheres to a code of conduct that expects all contributors to:

- Use welcoming and inclusive language
- Respect differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy toward other community members

## Development Environment Setup

### Prerequisites

- Python 3.9+
- Git
- Poetry (recommended) or pip

### Initial Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/figlet_forge.git
   cd figlet_forge
   ```

3. Set up a development environment:
   ```bash
   # Using Poetry (recommended)
   poetry install --with dev,test,docs

   # Using pip
   pip install -e ".[dev,test,docs]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Pull Request Process

1. **Branch Strategy**:
   - Create a new branch for your feature/fix:
     ```bash
     git checkout -b feature/description
     # or
     git checkout -b fix/issue-number-description
     ```

2. **Development Cycle**:
   - Write tests for your feature/fix
   - Implement your changes
   - Ensure all tests pass
   - Update documentation as needed

3. **Code Quality Checks**:
   - Run formatter:
     ```bash
     black .
     isort .
     ```
   - Run linters:
     ```bash
     flake8
     mypy .
     pylint .
     ```
   - Run tests:
     ```bash
     pytest
     ```

4. **Submit Pull Request**:
   - Push your branch to GitHub
   - Open a pull request against the `main` branch
   - Complete the pull request template

5. **Review Process**:
   - Address reviewer feedback
   - Maintain discussion etiquette
   - Make requested changes

## Style Guidelines

Figlet Forge adheres to strict Python styling conventions:

1. **Code Formatting**:
   - Black for code formatting
   - isort for import sorting
   - Maximum line length of 88 characters (Black default)

2. **Typing**:
   - Full type annotations for all functions and methods
   - Use type hints from `typing` module
   - Define custom types when appropriate

3. **Documentation**:
   - Google-style docstrings
   - Examples for complex functions
   - Markdown for rich documentation

4. **Naming Conventions**:
   - `snake_case` for variables, functions, and methods
   - `PascalCase` for classes
   - `UPPER_CASE` for constants
   - Avoid abbreviations, prefer clarity

Example:
```python
from typing import Dict, List, Optional

def process_figlet_text(
    text: str,
    font_name: Optional[str] = None,
    width: int = 80
) -> Dict[str, List[str]]:
    """
    Process text with figlet rendering.

    Args:
        text: The text to process
        font_name: Optional font name to use
        width: Maximum width of output

    Returns:
        Dictionary containing rendered lines and metadata

    Raises:
        FontNotFound: If the specified font cannot be found
    """
    # Implementation
```

## Testing Requirements

1. **Test Coverage**:
   - Minimum 90% code coverage
   - Unit tests for all public methods
   - Integration tests for complex interactions

2. **Test Organization**:
   - Unit tests in `tests/unit/`
   - Integration tests in `tests/integration/`
   - Compatibility tests in `tests/compat/`

3. **Test Execution**:
   - Tests must pass on all supported Python versions
   - Tests must pass on all supported platforms

## Documentation Standards

1. **API Documentation**:
   - Every public function/class must be documented
   - Include type information in docstrings
   - Provide examples for non-trivial usage

2. **User Documentation**:
   - Clear step-by-step tutorials
   - Comprehensive guides for features
   - Examples with outputs

3. **Visual Elements**:
   - Use ASCII diagrams for visual explanation
   - Include sample outputs for font examples
   - Use consistent formatting for code blocks

## Version Control Practices

1. **Commit Guidelines**:
   - Follow conventional commits format:
     ```
     type(scope): description
     ```
   - Types: feat, fix, docs, style, refactor, perf, test, chore
   - Keep commits focused and atomic
   - Write meaningful commit messages

2. **Branch Structure**:
   - `main`: Production-ready code
   - `develop`: Integration branch
   - `feature/*`: Feature development
   - `fix/*`: Bug fixes
   - `release/*`: Release preparation
   - `hotfix/*`: Urgent production fixes

## Release Process

1. **Versioning**:
   - Follow Semantic Versioning (MAJOR.MINOR.PATCH)
   - Document all changes in CHANGELOG.md

2. **Release Preparation**:
   - Create a release branch
   - Update version numbers
   - Update documentation
   - Finalize CHANGELOG.md
   - Create release candidate

3. **Testing**:
   - Comprehensive testing on all platforms
   - Verify documentation accuracy
   - Check backward compatibility

4. **Publishing**:
   - Merge to main branch
   - Create GitHub release
   - Publish to PyPI
   - Announce release

## Recognition

All contributors are recognized in the [CONTRIBUTORS.md](CONTRIBUTORS.md) file. Your contributions are greatly appreciated and will be properly attributed.

---

By contributing to Figlet Forge, you affirm that your contributions align with our licensing terms and Eidosian principles.

> _"Elegance in recursion, precision in expression."_ - Lloyd Handyside
