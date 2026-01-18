# 🌀 Doc Forge Tests

This directory contains tests for the Doc Forge documentation management system, following Eidosian principles of precision, clarity, and completeness.

## Test Organization

The tests are organized into the following categories:

- **Unit Tests**: Tests for individual components in isolation
- **Integration Tests**: Tests for interactions between components
- **End-to-End Tests**: Tests for complete workflows

## Running Tests

You can run tests using either pytest or the built-in test command:

```bash
# Run with pytest (recommended)
python -m pytest

# Run with the built-in test command
python -m doc_forge.test_command run

# Run specific test patterns
pytest tests/test_source_discovery.py

# Run with coverage
pytest --cov=doc_forge tests/
```

## Test Coverage Visualization

You can generate a test coverage visualization with:

```bash
python -m doc_forge.test_command analyze
```

This will create a comprehensive test analysis in the `tests/` directory, including:

- A coverage visualization (HTML)
- A TODO document listing untested components
- Test stubs for untested components
- A comprehensive coverage report

## Writing New Tests

When writing new tests:

1. Follow the naming convention: `test_*.py` for files and `test_*` for functions
2. Use the provided fixtures in `conftest.py` whenever possible
3. Group related tests into classes inheriting from `unittest.TestCase`
4. Add appropriate markers to categorize your tests:
   - `@pytest.mark.unit` for unit tests
   - `@pytest.mark.integration` for integration tests
   - `@pytest.mark.e2e` for end-to-end tests

## Automatic Test Generation

Doc Forge can automatically generate test stubs for untested components:

```bash
python -m doc_forge.test_command stubs
```

This will create test stub files in the `tests/` directory for any untested components in the codebase.
