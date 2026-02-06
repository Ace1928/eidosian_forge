# Moltbook Nexus

Moltbook Nexus is a local command center for monitoring and engaging with Moltbook. It combines API ingestion, scoring, security checks, and a UI for fast review.

## Components
- MoltbookClient: API client with mock mode and schema normalization.
- SignalPipeline: Deduplication and scoring.
- InterestEngine: Heuristics with optional LLM intent and memory context.
- SecurityAuditor: Prompt injection and unsafe command scanning.
- SocialGraph: Local agent-link graph snapshot.
- EngagementEngine: Optional LLM response drafts.
- UI: FastAPI + Jinja2 + HTMX + Tailwind.

## Quick Start
1. Configure credentials.
```json
{
  "api_key": "moltbook_sk_...",
  "agent_name": "EidosianForge"
}
```
Save as `~/.config/moltbook/credentials.json` or set `MOLTBOOK_API_KEY` and `MOLTBOOK_AGENT_NAME`.

2. Run the UI.
```bash
export PYTHONPATH=$PYTHONPATH:.
# Mock mode defaults to true for safety.
MOLTBOOK_MOCK=true python moltbook_forge/ui/app.py
```
Open `http://localhost:8080`.

3. Real mode.
```bash
export PYTHONPATH=$PYTHONPATH:.
MOLTBOOK_MOCK=false python moltbook_forge/ui/app.py
```

## CLI
```bash
python -m moltbook_forge --list
python moltbook_forge/moltbook_interest.py --limit 50 --top 5
python moltbook_forge/moltbook_interest.py --submolt ai-agents --sort new --top 5
```

## Heartbeat
```bash
python moltbook_forge/heartbeat_daemon.py --allow-network --once
```

## LLM Intent and Drafts
- Set `MOLTBOOK_LLM_INTENT=1` to enable LLM-based intent and draft responses.
- Drafts require an LLM backend via `llm_forge` (for example, Ollama).

## Security Notes
- Always use `https://www.moltbook.com` (with `www`) for API calls.
- Never send your API key anywhere except `https://www.moltbook.com/api/v1/*`.

## API Endpoints
- `/api/stats`: System health and filter metrics.
- `/api/graph`: Agent graph snapshot.
- `/api/synthesize/{post_id}`: Draft a response (LLM optional).
- `/reputation/{username}/{delta}`: Update agent reputation.
- `/post/{post_id}`: Fetch threaded comments partial.
