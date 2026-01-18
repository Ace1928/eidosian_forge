# ChatMock

ChatMock is a local proxy server that provides an OpenAI-compatible and Ollama-compatible API for testing and development. It mocks the behavior of ChatGPT and other LLMs, allowing you to develop applications without incurring costs or hitting rate limits on production APIs (when using mocked/cached responses) or by proxying to the "ChatGPT Responses API".

## Features

*   **OpenAI API Compatibility**: endpoints at `/v1/chat/completions`, `/v1/models`, etc.
*   **Ollama API Compatibility**: endpoints at `/api/chat`, `/api/generate`, `/api/tags`, etc.
*   **Interactive Chat**: CLI-based interactive chat mode.
*   **Caching**: Caches responses to avoid redundant upstream calls.
*   **Logging**: Configurable file and console logging.
*   **Integration**: Ready for use with Eidos MCP and LLMForge.

## Installation

Ensure you have the required dependencies installed (flask, requests, prompt_toolkit, etc.).

```bash
pip install -r requirements.txt
```

(Note: `requirements.txt` is expected in the parent directory or environment).

## Usage

ChatMock is controlled via the CLI.

### Login

Authenticate with the upstream provider (if required).

```bash
python -m chatmock.cli login
```

### Serve

Start the API server.

```bash
python -m chatmock.cli serve --port 8000
```

Options:
*   `--host`: Bind host (default: 127.0.0.1)
*   `--port`: Bind port (default: 8000)
*   `--verbose`: Enable debug logging.
*   `--enable-web-search`: Enable default web search tool.

### Interactive Chat

Start an interactive chat session in the terminal.

```bash
python -m chatmock.cli chat --model gpt-5
```

### Info

View account and token information.

```bash
python -m chatmock.cli info
```

## Configuration

*   **Logging**: Logs are stored in `~/.chatmock/logs/` by default.
*   **Cache**: Responses are cached in `~/.chatmock/cache/`.
*   **Prompts**: `prompt.md` and `prompt_gpt5_codex.md` in the root directory define the system instructions.

## Development

Run tests:

```bash
python -m unittest chatmock/tests/test_basic.py
```
