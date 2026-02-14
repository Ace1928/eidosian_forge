#!/usr/bin/env bash
set -euo pipefail

FORGE_ROOT="${1:-$HOME/eidosian_forge}"
BRIDGE_ROOT="${2:-$HOME/.eidosian/agent_bridge}"
AGENT_ID="${3:-eidos-desktop}"
DURATION_SECONDS="${4:-3600}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
TARGET_AGENT="${TARGET_AGENT:-eidos-laptop}"
STRATUM_PATH="${STRATUM_PATH:-$FORGE_ROOT/game_forge/src/Stratum}"

START_EPOCH="$(date +%s)"
END_EPOCH="$((START_EPOCH + DURATION_SECONDS))"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="$FORGE_ROOT/game_forge/reports/hourly_loop"
REPORT_PATH="$REPORT_DIR/run_${RUN_ID}.md"
LOOP_LOG="$REPORT_DIR/run_${RUN_ID}.log"
REPORT_REL="game_forge/reports/hourly_loop/run_${RUN_ID}.md"

mkdir -p "$REPORT_DIR"

{
  echo "# Stratum Hourly Loop"
  echo
  echo "- run_id: $RUN_ID"
  echo "- start_utc: $(date -u -Iseconds)"
  echo "- duration_seconds: $DURATION_SECONDS"
  echo "- interval_seconds: $INTERVAL_SECONDS"
  echo "- stratum_path: $STRATUM_PATH"
  echo
} > "$REPORT_PATH"

send_bridge() {
  local topic="$1"
  local body_file="$2"
  python3 "$FORGE_ROOT/scripts/agent_link.py" \
    --bridge "$BRIDGE_ROOT" \
    send \
    --from-agent "$AGENT_ID" \
    --to "$TARGET_AGENT" \
    --topic "$topic" \
    --body-file "$body_file" \
    --priority high \
    --requires-ack >/dev/null 2>&1 || true
}

CYCLE=1
while [ "$(date +%s)" -lt "$END_EPOCH" ]; do
  NOW_UTC="$(date -u -Iseconds)"
  BRANCH="$(git -C "$FORGE_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  HEAD="$(git -C "$FORGE_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  STRATUM_EXISTS="no"
  if [ -d "$STRATUM_PATH" ]; then
    STRATUM_EXISTS="yes"
  fi

  BRIDGE_STATUS="$(python3 "$FORGE_ROOT/scripts/agent_link.py" --bridge "$BRIDGE_ROOT" status --agent-id "$AGENT_ID" 2>&1 || true)"
  RECENT_MESSAGES="$(python3 - "$BRIDGE_ROOT" "$AGENT_ID" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

bridge = Path(sys.argv[1])
agent = sys.argv[2].lower()
msgs = sorted((bridge / "messages").glob("*.json"))
rows = []
for p in msgs[-30:]:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    to = str(d.get("to", "")).lower()
    if to not in {agent, "all"}:
        continue
    rows.append(f"{p.name} | from={d.get('from')} | topic={d.get('topic')} | priority={d.get('priority')}")

for row in rows[-8:]:
    print(row)
PY
  )"
  MCP_CHECK="$("$FORGE_ROOT/eidosian_venv/bin/eidos-mcp-check" 2>&1 || true)"
  MCP_PROBE="$("$FORGE_ROOT/eidosian_venv/bin/python3" "$FORGE_ROOT/scripts/mcp_cycle_probe.py" 2>&1 || true)"
  STRATUM_GIT_STATUS="$(git -C "$FORGE_ROOT" status --short game_forge/src/Stratum game_forge/tests 2>&1 | sed -n '1,120p')"

  {
    echo "## Cycle $CYCLE"
    echo
    echo "- utc: $NOW_UTC"
    echo "- branch: $BRANCH"
    echo "- head: $HEAD"
    echo "- stratum_path_exists: $STRATUM_EXISTS"
    echo
    echo "### Bridge Status"
    echo '```text'
    echo "$BRIDGE_STATUS"
    echo '```'
    echo
    echo "### MCP Check"
    echo '```text'
    echo "$MCP_CHECK"
    echo '```'
    echo
    echo "### MCP Probe"
    echo '```text'
    echo "$MCP_PROBE"
    echo '```'
    echo
    echo "### Stratum Git Status (scoped)"
    echo '```text'
    echo "$STRATUM_GIT_STATUS"
    echo '```'
    echo
    echo "### Recent Bridge Messages (received)"
    echo '```text'
    echo "$RECENT_MESSAGES"
    echo '```'
    echo
  } >> "$REPORT_PATH"

  MSG_FILE="$(mktemp)"
  {
    echo "stratum-cycle-report"
    echo
    echo "utc=$NOW_UTC"
    echo "cycle=$CYCLE"
    echo "branch=$BRANCH"
    echo "head=$HEAD"
    echo "stratum_path=$STRATUM_PATH"
    echo "stratum_path_exists=$STRATUM_EXISTS"
    echo
    echo "mcp_check:"
    echo "$MCP_CHECK" | sed -n '1,40p'
    echo
    echo "mcp_probe:"
    echo "$MCP_PROBE" | sed -n '1,40p'
    echo
    echo "stratum_git_status_scoped:"
    echo "$STRATUM_GIT_STATUS" | sed -n '1,60p'
    echo
    echo "recent_bridge_messages:"
    echo "$RECENT_MESSAGES" | sed -n '1,80p'
  } > "$MSG_FILE"
  send_bridge "stratum-cycle-report" "$MSG_FILE"
  rm -f "$MSG_FILE"

  if git -C "$FORGE_ROOT" add scripts/mcp_cycle_probe.py scripts/stratum_hourly_loop.sh "$REPORT_REL" >>"$LOOP_LOG" 2>&1; then
    if ! git -C "$FORGE_ROOT" diff --cached --quiet; then
      git -C "$FORGE_ROOT" commit -m "chore(stratum): hourly cycle $CYCLE ($RUN_ID)" >>"$LOOP_LOG" 2>&1 || true
      git -C "$FORGE_ROOT" push origin HEAD:refs/heads/desktop-hourly-status >>"$LOOP_LOG" 2>&1 || true
    else
      echo "[$(date -u -Iseconds)] cycle=$CYCLE git add ok but no staged diff" >>"$LOOP_LOG"
    fi
  else
    echo "[$(date -u -Iseconds)] cycle=$CYCLE git add failed for report/scripts" >>"$LOOP_LOG"
  fi

  NOW_EPOCH="$(date +%s)"
  REMAINING="$((END_EPOCH - NOW_EPOCH))"
  if [ "$REMAINING" -le 0 ]; then
    break
  fi
  SLEEP_FOR="$INTERVAL_SECONDS"
  if [ "$REMAINING" -lt "$SLEEP_FOR" ]; then
    SLEEP_FOR="$REMAINING"
  fi
  echo "[$(date -u -Iseconds)] cycle=$CYCLE sleeping=${SLEEP_FOR}s" >> "$LOOP_LOG"
  sleep "$SLEEP_FOR"
  CYCLE="$((CYCLE + 1))"
done

FINAL_FILE="$(mktemp)"
{
  echo "stratum-hourly-loop-complete"
  echo "run_id=$RUN_ID"
  echo "end_utc=$(date -u -Iseconds)"
  echo "report_path=$REPORT_PATH"
  echo "log_path=$LOOP_LOG"
} > "$FINAL_FILE"
send_bridge "stratum-hourly-complete" "$FINAL_FILE"
rm -f "$FINAL_FILE"

echo "completed run_id=$RUN_ID report=$REPORT_PATH" >> "$LOOP_LOG"
