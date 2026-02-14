#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sync_mesh_enroll.sh --node-name <name> [--role laptop|desktop] [--forge-root /home/lloyd/eidosian_forge]

Description:
  Prepares this host for the Eidos shared-mind sync stack:
  - Git for source-controlled code
  - Syncthing for selected state folders
  - Tailscale transport for LAN/WAN
EOF
}

NODE_NAME=""
ROLE="laptop"
FORGE_ROOT="/home/lloyd/eidosian_forge"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-name) NODE_NAME="${2:-}"; shift 2 ;;
    --role) ROLE="${2:-}"; shift 2 ;;
    --forge-root) FORGE_ROOT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$NODE_NAME" ]] || { echo "--node-name is required" >&2; exit 1; }
[[ -d "$FORGE_ROOT" ]] || { echo "Forge root not found: $FORGE_ROOT" >&2; exit 1; }

if command -v tailscale >/dev/null 2>&1; then
  systemctl enable --now tailscaled || true
fi

if command -v syncthing >/dev/null 2>&1; then
  systemctl --user enable --now syncthing.service || true
fi

SYNC_DIR="$HOME/.config/eidos-sync"
mkdir -p "$SYNC_DIR"

cat > "$SYNC_DIR/sync_mesh_manifest.v1.json" <<EOF
{
  "schema_version": "v1",
  "node_name": "${NODE_NAME}",
  "role": "${ROLE}",
  "forge_root": "${FORGE_ROOT}",
  "sync_folders": [
    "${FORGE_ROOT}/memory",
    "${FORGE_ROOT}/knowledge_cache",
    "${FORGE_ROOT}/data/shared_state"
  ],
  "excluded_patterns": [
    ".git/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "node_modules/",
    "*.log",
    "*.tmp",
    "*.cache",
    "models/",
    "SteamLibrary/"
  ]
}
EOF

mkdir -p "$FORGE_ROOT/data/shared_state"
cat > "$FORGE_ROOT/data/shared_state/.stignore" <<'EOF'
.git/
.venv/
venv/
__pycache__/
node_modules/
*.log
*.tmp
*.cache
models/
SteamLibrary/
EOF

echo "Sync mesh manifest: $SYNC_DIR/sync_mesh_manifest.v1.json"
echo "Next manual steps:"
echo "1) tailscale up"
echo "2) open Syncthing UI and add remote device IDs"
echo "3) set folder type for shared-mind folders (send/receive where required)"
