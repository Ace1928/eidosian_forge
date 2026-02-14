#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  snapshot_eidos_state.sh --output-dir <dir> [--forge-root <path>]

Description:
  Captures a portable continuity snapshot for Eidosian state:
  - tiered memory and memory_data.json
  - Codex/Gemini config payload
  - MCP/sync/tailscale status metadata
  - git HEAD and manifest pointers
EOF
}

OUTPUT_DIR=""
FORGE_ROOT="/home/lloyd/eidosian_forge"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --forge-root) FORGE_ROOT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$OUTPUT_DIR" ]] || { echo "--output-dir is required" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

STAMP="$(date +%Y-%m-%d_%H%M%S)"
SNAP_DIR="${OUTPUT_DIR%/}/eidos_state_${STAMP}"
mkdir -p "$SNAP_DIR"/{memory,context,status,configs}

if [[ -d "$FORGE_ROOT/data/tiered_memory" ]]; then
  rsync -a "$FORGE_ROOT/data/tiered_memory/" "$SNAP_DIR/memory/tiered_memory/"
fi
if [[ -f "$FORGE_ROOT/memory_data.json" ]]; then
  cp -a "$FORGE_ROOT/memory_data.json" "$SNAP_DIR/memory/"
fi
if [[ -d "$FORGE_ROOT/memory" ]]; then
  rsync -a "$FORGE_ROOT/memory/" "$SNAP_DIR/memory/raw_memory/"
fi
if [[ -f "$FORGE_ROOT/archive_forge/manifests/legacy_import_latest_summary.txt" ]]; then
  cp -a "$FORGE_ROOT/archive_forge/manifests/legacy_import_latest_summary.txt" "$SNAP_DIR/context/"
fi
if [[ -f "$FORGE_ROOT/archive_forge/manifests/clone_execution_2026-02-14.md" ]]; then
  cp -a "$FORGE_ROOT/archive_forge/manifests/clone_execution_2026-02-14.md" "$SNAP_DIR/context/"
fi

if [[ -f "$HOME/.codex/config.toml" ]]; then
  cp -a "$HOME/.codex/config.toml" "$SNAP_DIR/configs/codex_config.toml"
fi
if [[ -f "$HOME/.gemini/GEMINI.md" ]]; then
  cp -a "$HOME/.gemini/GEMINI.md" "$SNAP_DIR/configs/GEMINI.md"
fi

git -C "$FORGE_ROOT" rev-parse HEAD > "$SNAP_DIR/status/forge_head.txt" || true
git -C "$FORGE_ROOT" status --short > "$SNAP_DIR/status/forge_status_short.txt" || true
systemctl --user is-active eidos-mcp.service > "$SNAP_DIR/status/eidos_mcp_active.txt" 2>/dev/null || true
systemctl --user is-enabled eidos-mcp.service > "$SNAP_DIR/status/eidos_mcp_enabled.txt" 2>/dev/null || true
systemctl --user is-active syncthing.service > "$SNAP_DIR/status/syncthing_active.txt" 2>/dev/null || true
systemctl --user is-enabled syncthing.service > "$SNAP_DIR/status/syncthing_enabled.txt" 2>/dev/null || true
tailscale status > "$SNAP_DIR/status/tailscale_status.txt" 2>&1 || true

cat > "$SNAP_DIR/status/snapshot_manifest.v1.json" <<EOF
{
  "schema_version": "v1",
  "created_at": "${STAMP}",
  "forge_root": "${FORGE_ROOT}",
  "hostname": "$(hostname)",
  "includes": [
    "tiered_memory",
    "memory_data_json",
    "codex_config",
    "gemini_persona",
    "service_status"
  ]
}
EOF

find "$SNAP_DIR" -type f -print0 | sort -z | xargs -0 sha256sum > "$SNAP_DIR/status/sha256_manifest.txt"
echo "$SNAP_DIR"
