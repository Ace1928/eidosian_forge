# 🔥 LLM Forge

> A recursive framework for language model development, orchestration, and deployment.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)

## 📋 Overview

LLM Forge provides a sophisticated framework for working with language models through a recursive, self-optimizing approach. The system parses natural language prompts, generates structured content with multiple models, and applies recursive refinement to ensure high-quality, comprehensive responses.

```python
from llm_forge.response_loop import process_user_prompt

# Generate a structured response with multiple models and sections
response = process_user_prompt("Compare GPT and Claude models")
print(response["topic"])  # "GPT and Claude models"
```

## 🚀 Features

- **Structured Response Generation**: Transform natural language prompts into structured, multi-model responses
- **Recursive Self-Correction**: Automatic detection and completion of missing content
- **Type-Safe Architecture**: Comprehensive typing throughout for maximum reliability
- **Advanced Logging**: Contextual, color-coded logging with witty prefixes
- **Modular Design**: Easily extensible for custom model integration

## 💻 Installation

```bash
# Install from PyPI
pip install llm-forge

# Install from GitHub (development version)
pip install git+https://github.com/Ace1928/llm_forge.git

# Install with development dependencies
pip install llm-forge[dev]
```

## 🛠️ Usage

### Command Line Interface

```bash
# Generate a response to a prompt
llm-forge generate "Compare three different LLM architectures"

# Specify output file
llm-forge generate "Compare GPT-4 and Claude" --output comparison.json
```

### Python API

```python
from llm_forge.response_loop import process_user_prompt
from llm_forge.type_definitions import ModelResponse

# Generate structured response
user_prompt = "Compare three different LLM architectures: GPT, Claude, and Mistral"
response: ModelResponse = process_user_prompt(user_prompt)

# Access specific model and section
gpt_overview = response["models"]["gpt"]["overview"]
print(gpt_overview)
```

## 🧩 Architecture

LLM Forge follows a pipeline architecture:

1. **Input Parsing**: Transforms natural language prompts into structured requests
2. **Content Generation**: Creates content for each model and section
3. **Response Validation**: Ensures completeness of generated content
4. **Recursive Refinement**: Fills gaps and enhances quality

## 🧪 Development

```bash
# Clone the repository
git clone https://github.com/Ace1928/llm_forge.git
cd llm_forge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black .
isort .
ruff check .
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

- **Lloyd Handyside** - [Ace1928](https://github.com/Ace1928)

