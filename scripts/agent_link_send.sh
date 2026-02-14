#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <to> <topic> [body...]" >&2
  exit 1
fi

TO="$1"
TOPIC="$2"
shift 2
BODY="${*:-}"

python "$HOME/eidosian_forge/scripts/agent_link.py" \
  send \
  --from-agent "$(hostname)" \
  --to "$TO" \
  --topic "$TOPIC" \
  --body "$BODY" \
  --priority normal \
  --requires-ack
