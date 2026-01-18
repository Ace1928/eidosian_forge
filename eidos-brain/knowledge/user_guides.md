# End-User Guides

This document provides practical instructions for interacting with Eidos-Brain. Terminology is consistent with `glossary_reference.md`, while structure and snippets follow `templates.md` and `recursive_patterns.md`.

## CLI Commands

### Tutorial Application
Run the interactive tutorial to experiment with `EidosCore`:
```bash
python labs/tutorial_app.py [--load PATH] [--save PATH]
```
- `--load PATH` loads memories from a file before starting.
- `--save PATH` writes memories on exit.

### Glossary Generation
Generate the glossary of classes, functions, and constants:
```bash
python tools/generate_glossary.py
```
The glossary is written to `knowledge/glossary_reference.md`.

### Logbook Entry
Append a cycle entry to the project logbook:
```bash
python tools/logbook_entry.py "Summary" [--next TARGET] [--logbook PATH]
```
Use `--next` to note the next target for recursion.

## REST API Usage

A basic API exposes Eidos functionality via HTTP. The preferred implementation is `api.server`, which integrates FastAPI routes and provides WSGI helpers.

Start the server with:
```bash
uvicorn api.server:app --reload
```

### Example Requests
- `POST /remember` with JSON `{ "item": "data" }` stores a memory.
- `GET /memories` returns the current memory list.

### WSGI Deployment
Serve only the health endpoint with a WSGI host:
```bash
gunicorn api.server:create_app
```

## Deployment Instructions

1. Create and activate a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) run the tutorial or API as shown above.
4. For production API deployments, invoke `uvicorn` with a process manager such as `gunicorn`.
