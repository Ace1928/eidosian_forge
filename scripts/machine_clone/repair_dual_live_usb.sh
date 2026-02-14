#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sudo repair_dual_live_usb.sh \
    --device /dev/sdX \
    --ubuntu-iso /path/ubuntu.iso \
    --rescue-iso /path/systemrescue.iso \
    [--live-fs ext4|exfat] \
    [--casper-rw-size-gb 32] \
    [--forge-root /home/lloyd/eidosian_forge]

Description:
  Repairs the LIVE partition + boot config without touching encrypted vault (P3).
  - reformats partition 2 (LIVE_MULTI)
  - recopies Ubuntu + SystemRescue ISOs
  - recreates casper-rw persistence file
  - rewrites grub.cfg using build_dual_live_usb.sh renderer
  - runs verify_dual_live_usb.sh at the end
EOF
}

DEVICE=""
UBUNTU_ISO=""
RESCUE_ISO=""
LIVE_FS="ext4"
CASPER_RW_SIZE_GB="32"
FORGE_ROOT="/home/lloyd/eidosian_forge"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="${2:-}"; shift 2 ;;
    --ubuntu-iso) UBUNTU_ISO="${2:-}"; shift 2 ;;
    --rescue-iso) RESCUE_ISO="${2:-}"; shift 2 ;;
    --live-fs) LIVE_FS="${2:-}"; shift 2 ;;
    --casper-rw-size-gb) CASPER_RW_SIZE_GB="${2:-}"; shift 2 ;;
    --forge-root) FORGE_ROOT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "Run as root." >&2
  exit 1
fi

[[ -n "$DEVICE" ]] || { echo "--device is required" >&2; exit 1; }
[[ -n "$UBUNTU_ISO" ]] || { echo "--ubuntu-iso is required" >&2; exit 1; }
[[ -n "$RESCUE_ISO" ]] || { echo "--rescue-iso is required" >&2; exit 1; }
[[ -b "$DEVICE" ]] || { echo "Device not found: $DEVICE" >&2; exit 1; }
[[ -f "$UBUNTU_ISO" ]] || { echo "Ubuntu ISO not found: $UBUNTU_ISO" >&2; exit 1; }
[[ -f "$RESCUE_ISO" ]] || { echo "SystemRescue ISO not found: $RESCUE_ISO" >&2; exit 1; }
[[ "$CASPER_RW_SIZE_GB" =~ ^[0-9]+$ ]] || { echo "casper-rw-size-gb must be integer" >&2; exit 1; }
[[ "$LIVE_FS" == "ext4" || "$LIVE_FS" == "exfat" ]] || { echo "live-fs must be ext4 or exfat" >&2; exit 1; }

if [[ "$DEVICE" =~ [0-9]$ ]]; then
  P1="${DEVICE}p1"
  P2="${DEVICE}p2"
else
  P1="${DEVICE}1"
  P2="${DEVICE}2"
fi

[[ -b "$P1" ]] || { echo "Partition not found: $P1" >&2; exit 1; }
[[ -b "$P2" ]] || { echo "Partition not found: $P2" >&2; exit 1; }

for cmd in mkfs.ext4 mount umount truncate cp sync; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Missing command: $cmd" >&2; exit 1; }
done
if [[ "$LIVE_FS" == "exfat" ]]; then
  command -v mkfs.exfat >/dev/null 2>&1 || { echo "Missing command: mkfs.exfat" >&2; exit 1; }
fi

BUILD_SCRIPT="$FORGE_ROOT/scripts/machine_clone/build_dual_live_usb.sh"
VERIFY_SCRIPT="$FORGE_ROOT/scripts/machine_clone/verify_dual_live_usb.sh"
[[ -x "$BUILD_SCRIPT" ]] || { echo "Build script missing: $BUILD_SCRIPT" >&2; exit 1; }
[[ -x "$VERIFY_SCRIPT" ]] || { echo "Verify script missing: $VERIFY_SCRIPT" >&2; exit 1; }

while read -r mnt; do
  [[ -n "$mnt" ]] && umount -R "$mnt" || true
done < <(lsblk -nrpo MOUNTPOINTS "$DEVICE" | tr ',' '\n' | awk 'NF')

case "$LIVE_FS" in
  ext4) mkfs.ext4 -F -L LIVE_MULTI "$P2" ;;
  exfat) mkfs.exfat -n LIVE_MULTI "$P2" ;;
esac

TMP="$(mktemp -d)"
mkdir -p "$TMP/p1" "$TMP/p2"

cleanup() {
  set +e
  umount "$TMP/p2" >/dev/null 2>&1 || true
  umount "$TMP/p1" >/dev/null 2>&1 || true
  rmdir "$TMP/p2" "$TMP/p1" >/dev/null 2>&1 || true
  rmdir "$TMP" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mount "$P1" "$TMP/p1"
mount "$P2" "$TMP/p2"

cp -v --no-preserve=ownership,mode,timestamps "$UBUNTU_ISO" "$TMP/p2/ubuntu-live.iso"
cp -v --no-preserve=ownership,mode,timestamps "$RESCUE_ISO" "$TMP/p2/systemrescue.iso"

truncate -s "${CASPER_RW_SIZE_GB}G" "$TMP/p2/casper-rw"
mkfs.ext4 -F "$TMP/p2/casper-rw"

mkdir -p "$TMP/p2/eidos_bootstrap"
cp -v --no-preserve=ownership,mode,timestamps \
  "$FORGE_ROOT/scripts/machine_clone/bootstrap_eidos_machine.sh" \
  "$FORGE_ROOT/scripts/machine_clone/first_boot_wizard.sh" \
  "$FORGE_ROOT/scripts/machine_clone/README.md" \
  "$TMP/p2/eidos_bootstrap/" || true

mkdir -p "$TMP/p1/boot/grub"
"$BUILD_SCRIPT" --print-grub-config --live-fs "$LIVE_FS" > "$TMP/p1/boot/grub/grub.cfg"

sync
umount "$TMP/p2"
umount "$TMP/p1"

trap - EXIT
cleanup

"$VERIFY_SCRIPT" --device "$DEVICE"
echo "LIVE partition repair complete for $DEVICE"
