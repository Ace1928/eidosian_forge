#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  first_boot_wizard.sh [--forge-root <path>] [--interactive yes|no]
                       [--run-post-boot-check yes|no]
                       [--self-disable-autostart yes|no]

Description:
  Plan-first guided onboarding for restored Eidos machines.
  It inspects current status, builds an action plan with ETAs, asks for a single
  confirmation, then executes as much as possible automatically.
EOF
}

FORGE_ROOT="${HOME}/eidosian_forge"
INTERACTIVE="yes"
RUN_POST_BOOT_CHECK="yes"
SELF_DISABLE_AUTOSTART="no"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --forge-root) FORGE_ROOT="${2:-}"; shift 2 ;;
    --interactive) INTERACTIVE="${2:-}"; shift 2 ;;
    --run-post-boot-check) RUN_POST_BOOT_CHECK="${2:-}"; shift 2 ;;
    --self-disable-autostart) SELF_DISABLE_AUTOSTART="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

for v in "$INTERACTIVE" "$RUN_POST_BOOT_CHECK" "$SELF_DISABLE_AUTOSTART"; do
  [[ "$v" == "yes" || "$v" == "no" ]] || { echo "Boolean flags must be yes|no" >&2; exit 1; }
done

REPORT_DIR="${HOME}/.config/eidos-sync"
mkdir -p "$REPORT_DIR"
LOG_FILE="${REPORT_DIR}/first_boot_wizard_$(date +%Y%m%d_%H%M%S).log"
RUN_STARTED_AT="$(date +%s)"

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

declare -a PLAN_IDS=()
declare -a PLAN_DESC=()
declare -a PLAN_EST=()
declare -a PLAN_MODE=()   # auto|manual

format_seconds() {
  local s="${1:-0}"
  if (( s < 60 )); then
    printf "%ss" "$s"
    return
  fi
  local m=$((s / 60))
  local r=$((s % 60))
  printf "%sm%ss" "$m" "$r"
}

log_line() {
  echo "$*" | tee -a "$LOG_FILE"
}

ok() {
  PASS_COUNT=$((PASS_COUNT + 1))
  log_line "[OK] $*"
}

warn() {
  WARN_COUNT=$((WARN_COUNT + 1))
  log_line "[WARN] $*"
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  log_line "[FAIL] $*"
}

ask_yes_no() {
  local prompt="$1"
  local default="${2:-yes}"
  if [[ "$INTERACTIVE" != "yes" ]]; then
    [[ "$default" == "yes" ]]
    return
  fi
  local suffix="[Y/n]"
  [[ "$default" == "no" ]] && suffix="[y/N]"
  read -r -p "${prompt} ${suffix} " answer
  answer="${answer:-$default}"
  [[ "${answer,,}" == "y" || "${answer,,}" == "yes" ]]
}

run_cmd() {
  local cmd="$1"
  log_line "$ $cmd"
  bash -lc "$cmd" >>"$LOG_FILE" 2>&1
}

add_plan() {
  PLAN_IDS+=("$1")
  PLAN_DESC+=("$2")
  PLAN_EST+=("$3")
  PLAN_MODE+=("$4")
}

service_status() {
  local scope="$1"
  local service="$2"
  local enabled active
  if [[ "$scope" == "user" ]]; then
    enabled="$(systemctl --user is-enabled "$service" 2>/dev/null || echo unknown)"
    active="$(systemctl --user is-active "$service" 2>/dev/null || echo unknown)"
  else
    enabled="$(systemctl is-enabled "$service" 2>/dev/null || echo unknown)"
    active="$(systemctl is-active "$service" 2>/dev/null || echo unknown)"
  fi
  echo "${enabled}|${active}"
}

banner() {
  cat <<'EOF'
==============================================
  EIDOS FIRST-BOOT WIZARD (PLAN + EXECUTION)
==============================================
EOF
}

check_internet() {
  ping -c1 -W1 1.1.1.1 >/dev/null 2>&1
}

check_mcp_health() {
  curl -fsS "http://127.0.0.1:8928/health" >/tmp/.eidos_mcp_health.json 2>/dev/null
}

probe_mcp_ping() {
  local py="${FORGE_ROOT}/eidosian_venv/bin/python"
  [[ -x "$py" ]] || return 1
  "${py}" - <<'PY' >/tmp/.eidos_mcp_ping_probe.txt 2>/dev/null
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
async def main():
    async with streamable_http_client('http://127.0.0.1:8928/mcp') as (r,w,_):
        async with ClientSession(r,w) as s:
            await s.initialize()
            result = await s.call_tool('diagnostics_ping', arguments={})
            if result.structuredContent and 'result' in result.structuredContent:
                print(result.structuredContent['result'])
                return
            if result.content:
                for c in result.content:
                    if getattr(c, 'type', None) == 'text':
                        print(c.text)
                        return
            print("")
asyncio.run(main())
PY
}

probe_moltbook_api() {
  local creds="${HOME}/.config/moltbook/credentials.json"
  [[ -f "$creds" ]] || return 2
  command -v jq >/dev/null 2>&1 || return 3
  command -v curl >/dev/null 2>&1 || return 4
  local api_key
  api_key="$(jq -r '.api_key // ""' "$creds" 2>/dev/null || true)"
  [[ -n "$api_key" ]] || return 5
  local code
  code="$(curl -sS -o /tmp/.moltbook_probe.json -w '%{http_code}' \
    -H "Authorization: Bearer ${api_key}" \
    -H "X-API-Key: ${api_key}" \
    -H "Content-Type: application/json" \
    https://www.moltbook.com/api/v1/agents/status || true)"
  [[ "$code" == "200" ]]
}

run_plan_step() {
  local idx="$1"
  local total="$2"
  local step_no=$((idx + 1))
  local id="${PLAN_IDS[$idx]}"
  local desc="${PLAN_DESC[$idx]}"
  local est="${PLAN_EST[$idx]}"
  local mode="${PLAN_MODE[$idx]}"
  local elapsed now remaining
  now="$(date +%s)"
  elapsed=$((now - RUN_STARTED_AT))
  remaining=$((PLAN_TOTAL_EST - elapsed))
  (( remaining < 0 )) && remaining=0

  log_line
  log_line "[${step_no}/${total}] ${desc}"
  log_line "    mode=${mode} est=$(format_seconds "$est") elapsed=$(format_seconds "$elapsed") eta_remaining=$(format_seconds "$remaining")"

  local step_start rc=0
  step_start="$(date +%s)"

  case "$id" in
    create_venv)
      run_cmd "python3 -m venv '${FORGE_ROOT}/eidosian_venv'" || rc=$?
      ;;
    start_eidos_mcp)
      run_cmd "systemctl --user enable --now eidos-mcp.service" || rc=$?
      ;;
    start_syncthing)
      run_cmd "systemctl --user enable --now syncthing.service" || rc=$?
      ;;
    start_tailscaled)
      run_cmd "sudo systemctl enable --now tailscaled" || rc=$?
      ;;
    tailscale_auth)
      if ask_yes_no "Run tailscale auth now (browser flow required)?" yes; then
        run_cmd "sudo tailscale up --hostname=$(hostname -s) --accept-routes=false" || rc=$?
      else
        rc=20
      fi
      ;;
    fix_codex_config)
      run_cmd "mkdir -p '${HOME}/.codex'" || rc=$?
      if (( rc == 0 )); then
        if [[ -f "${HOME}/.codex/config.toml" ]]; then
          run_cmd "grep -q 'url = \"http://localhost:8928/mcp\"' '${HOME}/.codex/config.toml' || printf '\n[mcp_servers.eidosian_nexus]\nurl = \"http://localhost:8928/mcp\"\n' >> '${HOME}/.codex/config.toml'" || rc=$?
        else
          run_cmd "cat > '${HOME}/.codex/config.toml' <<'EOF'
model = \"gpt-5.3-codex\"
model_reasoning_effort = \"xhigh\"
personality = \"pragmatic\"

[mcp_servers.eidosian_nexus]
url = \"http://localhost:8928/mcp\"
EOF" || rc=$?
        fi
      fi
      ;;
    fix_moltbook_perms)
      run_cmd "chmod 600 '${HOME}/.config/moltbook/credentials.json'" || rc=$?
      ;;
    run_post_boot_check)
      run_cmd "'${FORGE_ROOT}/scripts/machine_clone/post_boot_onboarding.sh' --forge-root '${FORGE_ROOT}'" || rc=$?
      ;;
    *)
      rc=99
      ;;
  esac

  local step_elapsed
  step_elapsed=$(( $(date +%s) - step_start ))
  if (( rc == 0 )); then
    ok "completed ${id} in $(format_seconds "$step_elapsed")"
  elif (( rc == 20 )); then
    warn "skipped manual action ${id}"
  else
    warn "failed ${id} (exit=${rc}); see ${LOG_FILE}"
  fi
}

banner | tee -a "$LOG_FILE"
log_line "timestamp: $(date -Is)"
log_line "hostname:  $(hostname)"
log_line "forge:     ${FORGE_ROOT}"
log_line "log:       ${LOG_FILE}"

log_line
log_line "Current status snapshot:"

if check_internet; then
  ok "internet reachable"
else
  warn "internet ping not reachable yet"
fi

FORGE_EXISTS="no"
VENV_EXISTS="no"
if [[ -d "${FORGE_ROOT}/.git" ]]; then
  FORGE_EXISTS="yes"
  ok "forge repository present"
  log_line "branch: $(git -C "${FORGE_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
else
  fail "forge repository missing (${FORGE_ROOT})"
fi

if [[ -x "${FORGE_ROOT}/eidosian_venv/bin/python" ]]; then
  VENV_EXISTS="yes"
  ok "eidosian_venv present"
else
  warn "eidosian_venv missing"
  add_plan "create_venv" "Create eidosian_venv" 45 auto
fi

read -r emcp_enabled emcp_active <<<"$(service_status user eidos-mcp.service | tr '|' ' ')"
log_line "eidos-mcp.service enabled=${emcp_enabled} active=${emcp_active}"
if [[ "$emcp_active" != "active" ]]; then
  add_plan "start_eidos_mcp" "Start eidos-mcp.service" 8 auto
fi

read -r syn_enabled syn_active <<<"$(service_status user syncthing.service | tr '|' ' ')"
log_line "syncthing.service enabled=${syn_enabled} active=${syn_active}"
if [[ "$syn_active" != "active" ]]; then
  add_plan "start_syncthing" "Start syncthing.service" 8 auto
fi

read -r tails_enabled tails_active <<<"$(service_status system tailscaled | tr '|' ' ')"
log_line "tailscaled enabled=${tails_enabled} active=${tails_active}"
if [[ "$tails_active" != "active" ]]; then
  add_plan "start_tailscaled" "Start tailscaled system service" 10 auto
fi

TAILSCALE_BACKEND="unknown"
if command -v tailscale >/dev/null 2>&1; then
  TAILSCALE_BACKEND="$(tailscale status --json 2>/dev/null | jq -r '.BackendState // "unknown"' || echo "unknown")"
  log_line "tailscale backend=${TAILSCALE_BACKEND}"
  if [[ "$TAILSCALE_BACKEND" != "Running" ]]; then
    add_plan "tailscale_auth" "Authenticate tailscale node" 30 manual
  fi
else
  warn "tailscale command not found"
fi

if check_mcp_health; then
  ok "MCP health endpoint reachable"
  cat /tmp/.eidos_mcp_health.json | tee -a "$LOG_FILE" >/dev/null
else
  warn "MCP health endpoint not reachable yet"
fi
rm -f /tmp/.eidos_mcp_health.json

CODEX_CFG="${HOME}/.codex/config.toml"
if [[ -f "$CODEX_CFG" ]] && rg -q "url\\s*=\\s*\"http://localhost:8928/mcp\"" "$CODEX_CFG"; then
  ok "codex config points to local MCP"
else
  warn "codex config missing or MCP URL mismatch"
  add_plan "fix_codex_config" "Ensure codex MCP config points to localhost:8928/mcp" 3 auto
fi

MOLTBOOK_CREDS="${HOME}/.config/moltbook/credentials.json"
if [[ -f "$MOLTBOOK_CREDS" ]]; then
  perms="$(stat -c '%a' "$MOLTBOOK_CREDS" 2>/dev/null || echo unknown)"
  log_line "moltbook credentials perms=${perms}"
  if [[ "$perms" != "600" && "$perms" != "400" ]]; then
    add_plan "fix_moltbook_perms" "Tighten Moltbook credential file permissions" 2 auto
  fi
else
  warn "moltbook credentials missing"
fi

if [[ "$RUN_POST_BOOT_CHECK" == "yes" ]]; then
  add_plan "run_post_boot_check" "Run post_boot_onboarding diagnostics report" 20 auto
fi

PLAN_TOTAL_EST=0
for s in "${PLAN_EST[@]}"; do
  PLAN_TOTAL_EST=$((PLAN_TOTAL_EST + s))
done

log_line
log_line "Planned workflow:"
if (( ${#PLAN_IDS[@]} == 0 )); then
  log_line "  - No corrective actions required."
else
  for i in "${!PLAN_IDS[@]}"; do
    n=$((i + 1))
    log_line "  ${n}. ${PLAN_DESC[$i]} [mode=${PLAN_MODE[$i]} est=$(format_seconds "${PLAN_EST[$i]}")]"
  done
  log_line "  Total estimated execution time: $(format_seconds "$PLAN_TOTAL_EST")"
fi

EXECUTE_PLAN="yes"
if (( ${#PLAN_IDS[@]} > 0 )); then
  if ! ask_yes_no "Execute the full planned workflow automatically now?" yes; then
    EXECUTE_PLAN="no"
    warn "Plan execution cancelled by user"
  fi
fi

if [[ "$EXECUTE_PLAN" == "yes" && ${#PLAN_IDS[@]} -gt 0 ]]; then
  total_steps="${#PLAN_IDS[@]}"
  for i in "${!PLAN_IDS[@]}"; do
    run_plan_step "$i" "$total_steps"
  done
fi

log_line
log_line "Verification pass:"
if check_mcp_health; then
  ok "MCP health check passed after execution"
else
  fail "MCP health check still failing"
fi
rm -f /tmp/.eidos_mcp_health.json

if probe_mcp_ping; then
  if rg -q '^ok$' /tmp/.eidos_mcp_ping_probe.txt 2>/dev/null; then
    ok "MCP diagnostics_ping returned ok"
  else
    warn "MCP diagnostics_ping probe returned unexpected output"
  fi
else
  warn "MCP diagnostics_ping probe failed"
fi
rm -f /tmp/.eidos_mcp_ping_probe.txt

if probe_moltbook_api; then
  ok "Moltbook API credentials probe succeeded"
else
  warn "Moltbook API probe not successful (credentials/api may need attention)"
fi
rm -f /tmp/.moltbook_probe.json

TOTAL_ELAPSED=$(( $(date +%s) - RUN_STARTED_AT ))
log_line
log_line "=============================================="
log_line "Wizard complete"
log_line "pass=${PASS_COUNT} warn=${WARN_COUNT} fail=${FAIL_COUNT}"
log_line "elapsed=$(format_seconds "$TOTAL_ELAPSED")"
log_line "log=${LOG_FILE}"
log_line "=============================================="

if [[ "$SELF_DISABLE_AUTOSTART" == "yes" ]]; then
  AUTOSTART="${HOME}/.config/autostart/eidos-first-boot.desktop"
  if [[ -f "$AUTOSTART" ]]; then
    rm -f "$AUTOSTART"
    ok "Disabled one-shot autostart entry"
  fi
fi

exit 0
