#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  post_boot_onboarding.sh [--forge-root <path>] [--auto-tailscale-up yes|no]

Description:
  Runs practical post-boot checks for a newly restored Eidos machine and writes
  a report with exact next actions for tailscale, sync services, MCP, and venv.
EOF
}

FORGE_ROOT="${HOME}/eidosian_forge"
AUTO_TAILSCALE_UP="no"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --forge-root) FORGE_ROOT="${2:-}"; shift 2 ;;
    --auto-tailscale-up) AUTO_TAILSCALE_UP="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$AUTO_TAILSCALE_UP" != "yes" && "$AUTO_TAILSCALE_UP" != "no" ]]; then
  echo "--auto-tailscale-up must be yes|no" >&2
  exit 1
fi

REPORT_DIR="${HOME}/.config/eidos-sync"
mkdir -p "$REPORT_DIR"
REPORT_FILE="${REPORT_DIR}/post_boot_report_$(date +%Y%m%d_%H%M%S).txt"

log() {
  echo "$*" | tee -a "$REPORT_FILE"
}

status_or_unknown() {
  local cmd="$1"
  if eval "$cmd" >/dev/null 2>&1; then
    eval "$cmd" 2>/dev/null || true
  else
    echo "unknown"
  fi
}

log "Eidos Post-Boot Onboarding"
log "timestamp: $(date -Is)"
log "hostname: $(hostname)"
log "forge_root: ${FORGE_ROOT}"
log

if command -v tailscale >/dev/null 2>&1; then
  log "[tailscale]"
  systemctl is-enabled tailscaled >/dev/null 2>&1 || true
  systemctl is-active tailscaled >/dev/null 2>&1 || true
  log "tailscaled enabled: $(status_or_unknown 'systemctl is-enabled tailscaled')"
  log "tailscaled active:  $(status_or_unknown 'systemctl is-active tailscaled')"

  backend_state="$(tailscale status --json 2>/dev/null | jq -r '.BackendState // "unknown"' || echo "unknown")"
  log "backend state:      ${backend_state}"

  if [[ "$backend_state" != "Running" ]]; then
    tail_cmd="sudo tailscale up --hostname=$(hostname -s) --accept-routes=false"
    log "action required:    ${tail_cmd}"
    if [[ "$AUTO_TAILSCALE_UP" == "yes" ]]; then
      if sudo -n true >/dev/null 2>&1; then
        if $tail_cmd >>"$REPORT_FILE" 2>&1; then
          log "tailscale up:      executed successfully"
        else
          log "tailscale up:      failed; run command manually"
        fi
      else
        log "tailscale up:      skipped (sudo password required)"
      fi
    fi
  else
    log "tailscale status:   authenticated and running"
  fi
  log
fi

if command -v syncthing >/dev/null 2>&1; then
  log "[syncthing]"
  systemctl --user enable --now syncthing.service >/dev/null 2>&1 || true
  log "syncthing enabled:  $(status_or_unknown 'systemctl --user is-enabled syncthing.service')"
  log "syncthing active:   $(status_or_unknown 'systemctl --user is-active syncthing.service')"
  log
fi

log "[eidos-mcp]"
log "eidos-mcp enabled:  $(status_or_unknown 'systemctl --user is-enabled eidos-mcp.service')"
log "eidos-mcp active:   $(status_or_unknown 'systemctl --user is-active eidos-mcp.service')"
log

log "[forge git]"
if [[ -d "${FORGE_ROOT}/.git" ]]; then
  log "branch:             $(git -C "${FORGE_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  log "commit:             $(git -C "${FORGE_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  dirty_count="$(git -C "${FORGE_ROOT}" status --short | wc -l | tr -d ' ')"
  log "dirty files:        ${dirty_count}"
else
  log "forge repo missing at ${FORGE_ROOT}"
fi
log

log "[eidosian_venv]"
if [[ -x "${FORGE_ROOT}/eidosian_venv/bin/python" ]]; then
  log "venv python:        $("${FORGE_ROOT}/eidosian_venv/bin/python" --version 2>&1)"
  if "${FORGE_ROOT}/eidosian_venv/bin/python" -m pip check >>"$REPORT_FILE" 2>&1; then
    log "pip check:          ok"
  else
    log "pip check:          issues found (see report details)"
  fi
else
  log "venv missing:       ${FORGE_ROOT}/eidosian_venv"
  log "action required:    python3 -m venv ${FORGE_ROOT}/eidosian_venv"
fi
log

log "Verification commands:"
log "1) tailscale status"
log "2) systemctl --user status syncthing.service --no-pager"
log "3) systemctl --user status eidos-mcp.service --no-pager"
log "4) ${FORGE_ROOT}/eidosian_venv/bin/python -m pip check"
log
log "report: ${REPORT_FILE}"
