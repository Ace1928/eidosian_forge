#!/usr/bin/env bash
set -euo pipefail

AGENT_ID="${1:-$(hostname)}"
BRIDGE_ROOT="${2:-$HOME/.eidosian/agent_bridge}"
PY_BIN="${PY_BIN:-python}"

PID_FILE="$BRIDGE_ROOT/state/agent_link_${AGENT_ID}.pid"
LOG_FILE="$BRIDGE_ROOT/logs/agent_link_${AGENT_ID}.log"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "agent_link process: running pid=$PID"
  else
    echo "agent_link process: stale pid file"
  fi
else
  echo "agent_link process: not running"
fi

"$PY_BIN" "$HOME/eidosian_forge/scripts/agent_link.py" --bridge "$BRIDGE_ROOT" status --agent-id "$AGENT_ID"

if [[ -f "$LOG_FILE" ]]; then
  echo ""
  echo "log tail ($LOG_FILE):"
  tail -n 20 "$LOG_FILE" || true
fi
