#!/usr/bin/env bash
set -euo pipefail

AGENT_ID="${1:-$(hostname)}"
BRIDGE_ROOT="${2:-$HOME/.eidosian/agent_bridge}"
PY_BIN="${PY_BIN:-python}"

mkdir -p "$BRIDGE_ROOT/logs" "$BRIDGE_ROOT/state"

PID_FILE="$BRIDGE_ROOT/state/agent_link_${AGENT_ID}.pid"
LOG_FILE="$BRIDGE_ROOT/logs/agent_link_${AGENT_ID}.log"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "agent_link already running for $AGENT_ID (pid=$OLD_PID)"
    exit 0
  fi
fi

nohup "$PY_BIN" "$HOME/eidosian_forge/scripts/agent_link.py" \
  --bridge "$BRIDGE_ROOT" \
  watch \
  --agent-id "$AGENT_ID" \
  --ack \
  --only-unacked \
  --poll-interval 1.5 \
  --heartbeat-interval 5.0 \
  >> "$LOG_FILE" 2>&1 &

NEW_PID="$!"
echo "$NEW_PID" > "$PID_FILE"
echo "agent_link started for $AGENT_ID (pid=$NEW_PID)"
echo "log=$LOG_FILE"
