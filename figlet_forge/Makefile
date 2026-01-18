.PHONY: all clean minimal full publish test lint format check docs security venv tox develop install test-cov

all:
	@echo 'Figlet Forge: Enhanced Figlet System with colorized ANSI and Unicode support'
	@echo 'Run "make clean" to remove artifacts from previous builds'
	@echo 'Run "make minimal" to build a package with standard fonts only'
	@echo 'Run "make full" to build a complete package with all contributed fonts'
	@echo 'Run "make develop" to install package in development mode'
	@echo 'Run "make test" to run the test suite (use TEST_ARGS="..." for arguments)'
	@echo 'Run "make lint" to check code quality with ruff and mypy'
	@echo 'Run "make format" to format code according to project standards'
	@echo 'Run "make check" to run all pre-commit checks'
	@echo 'Run "make docs" to build documentation'
	@echo 'Run "make security" to run security scanners (bandit, safety)'
	@echo 'Run "make tox" to test across supported Python versions'
	@echo 'Run "make venv" to create a virtual environment for development'
	@echo 'Run "make publish" to upload to PyPI'

# Development installation - this works with the fixed setup.py
develop:
	@mkdir -p src/figlet_forge/fonts 2>/dev/null || true
	@if [ -d "src/figlet_forge/fonts-standard" ]; then \
		cp -r src/figlet_forge/fonts-standard/* src/figlet_forge/fonts/ 2>/dev/null || true; \
	fi
	pip install -e .

install:
	cp figlet_forge/fonts-standard/* figlet_forge/fonts/ 2>/dev/null || true
	pip install .

venv:
	python -m venv .venv
	@echo "Virtual environment created at .venv/"
	@echo "Activate with: source .venv/bin/activate"
	@echo "Then install with: make develop"
