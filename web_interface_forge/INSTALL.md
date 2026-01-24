# 🌐 Web Interface Forge Installation

Web UI components and server utilities.

## Quick Install

```bash
pip install -e ./web_interface_forge
python -c "from web_interface_forge import WebServer; print('✓ Ready')"
```

## CLI Usage

```bash
web-interface-forge serve --port 8080
web-interface-forge --help
```

## Dependencies

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `eidosian_core` - Decorators and logging

