# Development Guide

This document outlines the core standards for contributing to **LLM Forge**. It describes our coding conventions, docstring template, guiding design principles, and the process for maintaining the living glossary.

## Coding Standards

- **Formatting**: The codebase follows `black` (line length 88) and `isort` for import sorting. Run `black .` and `isort .` before committing.
- **Linting**: `ruff` enforces style rules. Execute `ruff check .` and address reported issues where possible.
- **Typing**: All public functions and methods must include Python type hints. `mypy` is used for optional static analysis.
- **Version Support**: The project targets Python 3.8+ as defined in `pyproject.toml`.
- **Tests**: Add unit tests in a `tests/` directory whenever adding new functionality. Use `pytest` and `pytest-cov`.

## Docstring Template

Every function, class, and module should contain a docstring structured as follows:

```python
"""One-line summary.

Optional extended description that provides context.

Args:
    param1: Explanation of the first parameter.
    param2: Explanation of the second parameter.

Returns:
    Description of the returned value.

Raises:
    ErrorType: Description of when the error occurs.

Examples:
    >>> example_call("argument")
"""
```

This template mirrors the existing style within the repository. Keep lines under 88 characters and prefer full sentences.

## Design Principles

- **Eidosian Recursion**: Components continually refine their output. Each processing step can invoke itself or other steps to ensure no ambiguity or inconsistency survives.
- **Modular Structure**: Place related functionality in clearly named modules. Each module should expose a minimal public API via `__all__` where appropriate.
- **Type Hints Everywhere**: Use precise typing for parameters, return values, and class attributes. Protocols and TypedDicts capture shared interfaces and data contracts.
- **Functional Tendencies**: Favor pure functions with explicit inputs and outputs. Minimize hidden state and side effects.
- **Self-Documenting Code**: Choose descriptive names, keep functions short, and include docstrings for every public element.

## Living Glossary

The repository maintains a *living glossary* of domain-specific terms inside `docs/GLOSSARY.md`. To add new terminology:

1. Append the term in alphabetical order with a short definition.
2. Submit a pull request describing the addition. If the term originates from new code or documentation, reference those files.
3. Keep definitions concise and link to further resources when relevant.

Expanding the glossary ensures that the language used across documentation and code remains precise and accessible.

