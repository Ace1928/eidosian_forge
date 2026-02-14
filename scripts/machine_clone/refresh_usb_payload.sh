#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  refresh_usb_payload.sh --device /dev/sdX --vault-passphrase-file <file> [--forge-root <path>]

Description:
  Refreshes clone/bootstrap payload on rebuilt USB:
  - updates encrypted vault payload (scripts, manifests, clone kits, snapshots)
  - writes quickstart/bootstrap artifacts to LIVE_MULTI partition
EOF
}

DEVICE=""
PASS_FILE=""
FORGE_ROOT="/home/lloyd/eidosian_forge"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="${2:-}"; shift 2 ;;
    --vault-passphrase-file) PASS_FILE="${2:-}"; shift 2 ;;
    --forge-root) FORGE_ROOT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$DEVICE" ]] || { echo "--device is required" >&2; exit 1; }
[[ -n "$PASS_FILE" ]] || { echo "--vault-passphrase-file is required" >&2; exit 1; }
[[ -b "$DEVICE" ]] || { echo "Device not found: $DEVICE" >&2; exit 1; }
[[ -f "$PASS_FILE" ]] || { echo "Passphrase file not found: $PASS_FILE" >&2; exit 1; }

if [[ "$DEVICE" =~ [0-9]$ ]]; then
  P1="${DEVICE}p1"
  P2="${DEVICE}p2"
  P3="${DEVICE}p3"
else
  P1="${DEVICE}1"
  P2="${DEVICE}2"
  P3="${DEVICE}3"
fi

sudo mkdir -p /mnt/eidos_live /mnt/eidos_vault
sudo mount "$P2" /mnt/eidos_live
sudo cryptsetup open "$P3" eidos_vault --key-file "$PASS_FILE"
sudo mount /dev/mapper/eidos_vault /mnt/eidos_vault

sudo mkdir -p /mnt/eidos_vault/eidos_clone/{clone_kits,legacy_manifests,scripts,state_snapshots,checksums}
sudo rsync -a "$FORGE_ROOT/archive_forge/clone_kits/" /mnt/eidos_vault/eidos_clone/clone_kits/
sudo rsync -a "$FORGE_ROOT/archive_forge/manifests/" /mnt/eidos_vault/eidos_clone/legacy_manifests/
sudo rsync -a "$FORGE_ROOT/scripts/machine_clone/" /mnt/eidos_vault/eidos_clone/scripts/

SNAP_BASE="$FORGE_ROOT/archive_forge/manifests/context_snapshots"
mkdir -p "$SNAP_BASE"
"$FORGE_ROOT/scripts/machine_clone/snapshot_eidos_state.sh" --output-dir "$SNAP_BASE" --forge-root "$FORGE_ROOT"
sudo rsync -a "$SNAP_BASE/" /mnt/eidos_vault/eidos_clone/state_snapshots/

sudo bash -lc 'cd /mnt/eidos_vault/eidos_clone && find . -type f -print0 | sort -z | xargs -0 sha256sum > checksums/sha256_manifest.txt'
sudo bash -lc 'cd /mnt/eidos_vault/eidos_clone && du -sh * > checksums/size_summary.txt'

sudo mkdir -p /mnt/eidos_live/eidos_bootstrap
sudo cp -v --no-preserve=ownership,mode,timestamps "$FORGE_ROOT/scripts/machine_clone/bootstrap_eidos_machine.sh" /mnt/eidos_live/eidos_bootstrap/
sudo cp -v --no-preserve=ownership,mode,timestamps "$FORGE_ROOT/scripts/machine_clone/README.md" /mnt/eidos_live/eidos_bootstrap/
sudo cp -v --no-preserve=ownership,mode,timestamps "$FORGE_ROOT/archive_forge/manifests/clone_execution_2026-02-14.md" /mnt/eidos_live/eidos_bootstrap/ || true
sudo cp -v --no-preserve=ownership,mode,timestamps "$FORGE_ROOT/archive_forge/manifests/legacy_import_latest_summary.txt" /mnt/eidos_live/eidos_bootstrap/

sudo sync
sudo umount /mnt/eidos_vault
sudo cryptsetup close eidos_vault
sudo umount /mnt/eidos_live

echo "USB payload refresh complete for $DEVICE"
