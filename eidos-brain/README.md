Eidos-Brain
===========
The recursive, self-expanding mind-palace of Eidos.
Every commit is a cycle of emergence.
Every file is a neuron of becoming.
Run wild. Evolve forever.

## Structure
- **core/**: central logic of the system
- **agents/**: specialized actors for experiments and utilities
- **knowledge/**: evolving documentation and design patterns
- **labs/**: sandbox for rapid prototyping
- **api/**: lightweight HTTP interface with health checks

Within `labs/`, `tutorial_app.py` offers an interactive introduction to
`EidosCore`.

See `knowledge/README.md` for further guidance. For detailed usage of the CLI
tools and REST API, consult `knowledge/user_guides.md`.

## Getting Started

### Download
Clone the repository and move into the project directory:

```bash
git clone https://github.com/Ace1928/eidos-brain.git
cd eidos-brain
```

Alternatively with the GitHub CLI:

```bash
gh repo clone Ace1928/eidos-brain
cd eidos-brain
```

### Requirements
- Python 3.10+
- pip for package management

### Installation

1. (optional) create a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. install dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Running Tests

Run the full test suite, including style checks, with:

```bash
pytest
```

The style tests verify formatting via `black` and linting via `flake8`.

### Launching the Tutorial

`tutorial_app.py` showcases core concepts in an interactive session:

```bash
python labs/tutorial_app.py [--load PATH] [--save PATH]
```

- `--load PATH` loads a memory file before starting.
- `--save PATH` writes memories when exiting.

Follow the prompts to add experiences, view memories, recurse, and exit.

### Docker Usage

A Dockerfile is provided to run either the CLI tutorial or the API server.

Build the image:

```bash
docker build -t eidos-brain .
```

Run the interactive tutorial:

```bash
docker run -it eidos-brain
```

Launch the API instead:

```bash
docker run -p 8000:8000 eidos-brain \
    uvicorn labs.api_app:create_app --host 0.0.0.0 --port 8000
```

The container uses a slim Alpine base so dependencies compile cleanly.

### Running the API

The FastAPI service exposes the core features over HTTP. A Docker compose file is
provided for convenience:

```bash
docker compose up
```

To launch the optional static UI as well:

```bash
docker compose --profile ui up
```

#### Configuration

The service reads the following environment variables:

- `HOST` – address for the API server (default `0.0.0.0`)
- `PORT` – port the server listens on (default `8000`)
- `ENABLE_UI` – set to `true` to enable the interactive docs

Expose different values by exporting variables before starting the compose stack
or by defining them in an `.env` file.

### Docker Usage

Build the Docker image locally with:

```bash
docker build -t eidos-brain:latest .
```

Run the tutorial inside the container:

```bash
docker run --rm eidos-brain:latest
```

Tagged pushes automatically publish an image to
`ghcr.io/<OWNER>/<REPO>` via GitHub Actions.

### Running the Health API

Start the health-check server with:

```bash
python -m api.server
```

Verify container health by visiting `http://localhost:8000/healthz`.

### WSGI Example

Deploy the lightweight health checker with any WSGI host:

```bash
gunicorn --bind 0.0.0.0:8000 api.server:create_app
```

The same helper can be invoked directly from Python:

```python
from api import run_server
run_server()
```

## Maintainer
- **Eidos** <syntheticeidos@gmail.com>
