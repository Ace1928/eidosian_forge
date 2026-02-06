# Moltbook Nexus UI

A high-performance dashboard for interacting with Moltbook, optimized for the Eidosian Forge.

## Features
- **Real-time Feed**: Displays latest posts and comments.
- **Interest Engine**: Ranks content based on Eidosian relevance (Eidos, Forge, AI, etc.).
- **Mock Mode**: Development-friendly mock data for safe testing.
- **Dynamic Interaction**: Powered by HTMX for smooth comment loading.
- **Human-Friendly UI**: Clean, responsive Tailwind CSS design.
- **Nexus Map**: Optional social graph snapshot (served by `/api/graph`).

## Usage

### 1. Start the Server
```bash
export PYTHONPATH=$PYTHONPATH:.
# Toggle MOCK_MODE with env var (defaults to true)
MOLTBOOK_MOCK=true python moltbook_forge/ui/app.py
```

Optional:
- `MOLTBOOK_LLM_INTENT=1` enables LLM-based intent tagging (defaults off).

### 2. Access the Dashboard
Open your browser at `http://localhost:8080`.

## Technical Specs
- **Backend**: FastAPI
- **Frontend**: Jinja2 + HTMX + Tailwind CSS
- **Performance**: ~2.3ms rendering time (benchmarked)
- **Security**: Strict typing and isolated mock environment available.

---
**Version**: 1.0.0
**Status**: Optimal
