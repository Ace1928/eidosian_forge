#!/usr/bin/env bash
set -euo pipefail

AGENT_ID="${1:-$(hostname)}"
BRIDGE_ROOT="${2:-$HOME/.eidosian/agent_bridge}"

PID_FILE="$BRIDGE_ROOT/state/agent_link_${AGENT_ID}.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No pid file for agent_link ($AGENT_ID)."
  exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "${PID:-}" ]]; then
  rm -f "$PID_FILE"
  echo "Empty pid file removed."
  exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  sleep 1
  if kill -0 "$PID" 2>/dev/null; then
    kill -9 "$PID" || true
  fi
  echo "agent_link stopped for $AGENT_ID (pid=$PID)"
else
  echo "Process not running (pid=$PID), cleaning pid file."
fi

rm -f "$PID_FILE"
