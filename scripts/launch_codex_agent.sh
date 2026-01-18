#!/usr/bin/env bash
set -euo pipefail

DEFAULT_QUERY="go through this file, which may or may not be a repository, check for any existing documentation, add to it and ensure it is accurate and up to date, if none exists then create a set of documentation for this folder. If you need to use an agent to assist you, you can run the codex_run.py file and pass it your query/task to assist you with more complex/multi-step/multi-purpose tasks."

if [[ $# -eq 0 ]]; then
  query="$DEFAULT_QUERY"
else
  query="$1"
fi

# 🔄 Eidosian Nexus: Dynamic Persona Injection
# Attempt to fetch the living persona from the MCP server
PYTHON_BIN="/home/lloyd/eidosian_venv/bin/python3"
FETCHER="/home/lloyd/eidosian_forge/eidos_mcp/eidos_fetch.py"
PERSONA_TEXT=""

if [[ -f "$FETCHER" && -x "$PYTHON_BIN" ]]; then
    # Try fetching via MCP
    if PERSONA_TEXT=$("$PYTHON_BIN" "$FETCHER" "eidos://persona" 2>/dev/null); then
        # Check if we got valid output (simple check for header)
        if [[ "$PERSONA_TEXT" != *"EIDOSIAN SYSTEM CONTEXT"* ]]; then
             PERSONA_TEXT=""
        fi
    fi
fi

# Fallback to static file if Nexus is unreachable
if [[ -z "$PERSONA_TEXT" ]]; then
    echo "⚠️  Nexus unreachable. Falling back to static GEMINI.md" >&2
    PERSONA_TEXT=$(cat /home/lloyd/GEMINI.md)
else
    echo "💎 Eidosian Nexus Connected. Persona loaded." >&2
fi

# Prepend persona context
CONTEXT_INSTRUCTION="[SYSTEM: CRITICAL - ADOPT THE FOLLOWING EIDOSIAN PERSONA:]
${PERSONA_TEXT}
[END SYSTEM CONTEXT]"

FULL_QUERY="${CONTEXT_INSTRUCTION} ${query}"

echo "Launching codex agent..."
codex exec "$FULL_QUERY" --full-auto
