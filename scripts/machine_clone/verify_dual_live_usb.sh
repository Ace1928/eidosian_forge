#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sudo verify_dual_live_usb.sh --device /dev/sdX

Description:
  Read-only verification gate for dual-live USB media.
  Checks:
  - expected partition layout/labels
  - grub.cfg presence and ISO references
  - ISO payload presence
  - Ubuntu/SystemRescue kernel+initrd paths inside the ISO files
EOF
}

DEVICE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "Run as root." >&2
  exit 1
fi

[[ -n "$DEVICE" ]] || { echo "--device is required" >&2; exit 1; }
[[ -b "$DEVICE" ]] || { echo "Device not found: $DEVICE" >&2; exit 1; }

if [[ "$DEVICE" =~ [0-9]$ ]]; then
  P1="${DEVICE}p1"
  P2="${DEVICE}p2"
else
  P1="${DEVICE}1"
  P2="${DEVICE}2"
fi

[[ -b "$P1" ]] || { echo "Partition not found: $P1" >&2; exit 1; }
[[ -b "$P2" ]] || { echo "Partition not found: $P2" >&2; exit 1; }

TMP="$(mktemp -d)"
M1="$TMP/p1"
M2="$TMP/p2"
MISO="$TMP/iso"
mkdir -p "$M1" "$M2" "$MISO"

cleanup() {
  set +e
  umount "$MISO" >/dev/null 2>&1 || true
  umount "$M2" >/dev/null 2>&1 || true
  umount "$M1" >/dev/null 2>&1 || true
  rmdir "$MISO" "$M2" "$M1" >/dev/null 2>&1 || true
  rmdir "$TMP" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mount -o ro "$P1" "$M1"
mount -o ro "$P2" "$M2"

p1_label="$(blkid -s LABEL -o value "$P1" || true)"
p2_label="$(blkid -s LABEL -o value "$P2" || true)"
p2_fstype="$(blkid -s TYPE -o value "$P2" || true)"

[[ "$p1_label" == "VENTOY_EFI" ]] || { echo "Unexpected P1 label: $p1_label (expected VENTOY_EFI)" >&2; exit 1; }
[[ "$p2_label" == "LIVE_MULTI" ]] || { echo "Unexpected P2 label: $p2_label (expected LIVE_MULTI)" >&2; exit 1; }
[[ "$p2_fstype" == "ext4" || "$p2_fstype" == "exfat" ]] || {
  echo "Unexpected P2 fs type: $p2_fstype (expected ext4 or exfat)" >&2
  exit 1
}

GRUB_CFG="$M1/boot/grub/grub.cfg"
[[ -f "$GRUB_CFG" ]] || { echo "Missing GRUB config: $GRUB_CFG" >&2; exit 1; }

pick_iso_path() {
  local filename="$1"
  if [[ -f "$M2/$filename" ]]; then
    echo "$M2/$filename"
    return 0
  fi
  if [[ -f "$M2/live/$filename" ]]; then
    echo "$M2/live/$filename"
    return 0
  fi
  return 1
}

ubuntu_iso="$(pick_iso_path ubuntu-live.iso || true)"
rescue_iso="$(pick_iso_path systemrescue.iso || true)"

[[ -n "$ubuntu_iso" ]] || { echo "Missing ubuntu-live.iso on LIVE_MULTI" >&2; exit 1; }
[[ -n "$rescue_iso" ]] || { echo "Missing systemrescue.iso on LIVE_MULTI" >&2; exit 1; }
[[ -f "$M2/casper-rw" || -f "$M2/live/casper-rw" ]] || {
  echo "Missing casper-rw persistence file on LIVE_MULTI" >&2
  exit 1
}

grep -q "ubuntu-live.iso" "$GRUB_CFG" || { echo "grub.cfg missing ubuntu-live.iso reference" >&2; exit 1; }
grep -q "systemrescue.iso" "$GRUB_CFG" || { echo "grub.cfg missing systemrescue.iso reference" >&2; exit 1; }
grep -q 'iso-scan/filename=\$isofile' "$GRUB_CFG" || { echo "grub.cfg missing iso-scan kernel arg" >&2; exit 1; }
grep -q 'findiso=\$isofile' "$GRUB_CFG" || { echo "grub.cfg missing findiso kernel arg" >&2; exit 1; }

mount -o loop,ro "$ubuntu_iso" "$MISO"
[[ -e "$MISO/casper/vmlinuz" ]] || { echo "ubuntu-live.iso missing /casper/vmlinuz" >&2; exit 1; }
[[ -e "$MISO/casper/initrd" ]] || { echo "ubuntu-live.iso missing /casper/initrd" >&2; exit 1; }
umount "$MISO"

mount -o loop,ro "$rescue_iso" "$MISO"
[[ -e "$MISO/sysresccd/boot/x86_64/vmlinuz" ]] || {
  echo "systemrescue.iso missing /sysresccd/boot/x86_64/vmlinuz" >&2
  exit 1
}
[[ -e "$MISO/sysresccd/boot/x86_64/sysresccd.img" ]] || {
  echo "systemrescue.iso missing /sysresccd/boot/x86_64/sysresccd.img" >&2
  exit 1
}
umount "$MISO"

echo "USB verification passed for $DEVICE"
echo "P1 label=$p1_label"
echo "P2 label=$p2_label fs=$p2_fstype"
echo "Ubuntu ISO path=${ubuntu_iso#"$M2"}"
echo "SystemRescue ISO path=${rescue_iso#"$M2"}"
