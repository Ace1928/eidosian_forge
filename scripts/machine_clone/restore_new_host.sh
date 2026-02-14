#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  restore_new_host.sh --profile <host_profile_dir> [--mode hybrid] [--apply-secrets <path>] [--forge-root <path>] [--restore-memory yes|no] [--setup-eidos-mcp yes|no] [--setup-codex yes|no]

Description:
  Restores package manifests, desktop configuration, Eidosian continuity state,
  Codex configuration, and MCP service wiring from an exported host profile.
EOF
}

PROFILE=""
MODE="hybrid"
SECRETS_ARCHIVE=""
FORGE_ROOT="${HOME}/eidosian_forge"
RESTORE_MEMORY="yes"
SETUP_EIDOS_MCP="yes"
SETUP_CODEX="yes"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --apply-secrets) SECRETS_ARCHIVE="${2:-}"; shift 2 ;;
    --forge-root) FORGE_ROOT="${2:-}"; shift 2 ;;
    --restore-memory) RESTORE_MEMORY="${2:-}"; shift 2 ;;
    --setup-eidos-mcp) SETUP_EIDOS_MCP="${2:-}"; shift 2 ;;
    --setup-codex) SETUP_CODEX="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$PROFILE" ]] || { echo "--profile is required" >&2; exit 1; }
[[ -d "$PROFILE" ]] || { echo "Profile dir not found: $PROFILE" >&2; exit 1; }
[[ "$MODE" == "hybrid" ]] || { echo "Unsupported mode: $MODE" >&2; exit 1; }

PACKAGES_DIR="$PROFILE/packages"
CONFIGS_DIR="$PROFILE/configs"
ASSETS_DIR="$PROFILE/assets"

if [[ $EUID -eq 0 ]]; then
  if [[ -f "$PACKAGES_DIR/dpkg-selections.txt" ]]; then
    dpkg --set-selections < "$PACKAGES_DIR/dpkg-selections.txt"
    apt-get update
    apt-get -y dselect-upgrade || true
  fi
else
  echo "Not root: skipping apt restore. Re-run with sudo for package restore." >&2
fi

if command -v snap >/dev/null 2>&1 && [[ -f "$PACKAGES_DIR/snap-list.txt" ]]; then
  awk 'NR>1 {print $1}' "$PACKAGES_DIR/snap-list.txt" | while read -r s; do
    [[ -n "$s" ]] || continue
    snap list "$s" >/dev/null 2>&1 || snap install "$s" || true
  done
fi

if [[ -d "$CONFIGS_DIR/home" ]]; then
  rsync -a "$CONFIGS_DIR/home/" "$HOME/"
fi

# Harden secrets-bearing client credentials restored from profile sync.
if [[ -f "$HOME/.config/moltbook/credentials.json" ]]; then
  chmod 600 "$HOME/.config/moltbook/credentials.json" || true
fi

if [[ -f "$CONFIGS_DIR/dconf_dump.ini" ]]; then
  dconf load / < "$CONFIGS_DIR/dconf_dump.ini" || true
fi

if [[ -d "$ASSETS_DIR/home-pictures" ]]; then
  mkdir -p "$HOME/Pictures"
  rsync -a "$ASSETS_DIR/home-pictures/" "$HOME/Pictures/"
fi

if [[ -d "$ASSETS_DIR/home-fonts" ]]; then
  mkdir -p "$HOME/.local/share/fonts"
  rsync -a "$ASSETS_DIR/home-fonts/" "$HOME/.local/share/fonts/"
fi

if [[ "$RESTORE_MEMORY" == "yes" && -d "$ASSETS_DIR/eidosian_forge_state" && -d "$FORGE_ROOT" ]]; then
  rsync -a "$ASSETS_DIR/eidosian_forge_state/" "$FORGE_ROOT/"
fi

if [[ "$SETUP_CODEX" == "yes" ]]; then
  mkdir -p "$HOME/.codex"
  if [[ -f "$HOME/.codex/config.toml" ]]; then
    chmod 600 "$HOME/.codex/config.toml" || true
  else
    cat > "$HOME/.codex/config.toml" <<'EOF'
model = "gpt-5.3-codex"
model_reasoning_effort = "xhigh"
personality = "pragmatic"

[mcp_servers.eidosian_nexus]
url = "http://localhost:8928/mcp"
EOF
    chmod 600 "$HOME/.codex/config.toml"
  fi
fi

if [[ "$SETUP_EIDOS_MCP" == "yes" && -d "$FORGE_ROOT/eidos_mcp" ]]; then
  mkdir -p "$HOME/.config/systemd/user"
  cat > "$HOME/.config/systemd/user/eidos-mcp.service" <<EOF
[Unit]
Description=Eidosian MCP Server (Streamable HTTP)
After=network.target

[Service]
Type=simple
WorkingDirectory=${FORGE_ROOT}/eidos_mcp
Environment=EIDOS_FORGE_DIR=${FORGE_ROOT}
Environment=EIDOS_MCP_TRANSPORT=streamable-http
Environment=EIDOS_MCP_MOUNT_PATH=/mcp
Environment=EIDOS_MCP_STATELESS_HTTP=1
Environment=EIDOS_MCP_ENABLE_COMPAT_HEADERS=1
Environment=EIDOS_MCP_ENABLE_SESSION_RECOVERY=1
Environment=EIDOS_MCP_ENABLE_ERROR_RESPONSE_COMPAT=1
Environment=FASTMCP_HOST=127.0.0.1
Environment=FASTMCP_PORT=8928
Environment=EIDOS_MCP_SERVICE_STATE=%h/.eidosian/run/mcp_service_state.json
Environment=EIDOS_MCP_BACKUP_DIR=%h/.eidosian/backups/eidos_mcp
ExecStart=${FORGE_ROOT}/eidosian_venv/bin/python3 -m eidos_mcp.eidos_mcp_server
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now eidos-mcp.service || true
fi

if [[ -n "$SECRETS_ARCHIVE" ]]; then
  echo "Secrets archive provided: $SECRETS_ARCHIVE"
  echo "Decrypt and apply manually to avoid accidental overwrite."
fi

echo "Restore workflow complete."
