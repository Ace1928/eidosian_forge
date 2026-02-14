#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sudo build_dual_live_usb.sh \
    --device /dev/sdX \
    --ubuntu-iso /path/ubuntu.iso \
    --rescue-iso /path/systemrescue.iso \
    [--live-size-gb 126] \
    [--vault-passphrase-file /path/passphrase.txt]

Description:
  Creates a GPT layout with:
  - partition 1: FAT32 EFI boot (2GiB)
  - partition 2: exFAT LIVE_MULTI for ISO payloads and persistence file
  - partition 3: LUKS2 + ext4 EIDOS_VAULT

  Installs GRUB EFI to the USB and writes loopback boot entries for both ISOs.
EOF
}

DEVICE=""
UBUNTU_ISO=""
RESCUE_ISO=""
LIVE_SIZE_GB="126"
VAULT_PASS_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="${2:-}"; shift 2 ;;
    --ubuntu-iso) UBUNTU_ISO="${2:-}"; shift 2 ;;
    --rescue-iso) RESCUE_ISO="${2:-}"; shift 2 ;;
    --live-size-gb) LIVE_SIZE_GB="${2:-}"; shift 2 ;;
    --vault-passphrase-file) VAULT_PASS_FILE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "Run as root." >&2
  exit 1
fi

for cmd in parted sgdisk wipefs mkfs.vfat mkfs.exfat cryptsetup grub-install rsync; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Missing command: $cmd" >&2; exit 1; }
done

if [[ -z "$DEVICE" || -z "$UBUNTU_ISO" || -z "$RESCUE_ISO" ]]; then
  echo "--device, --ubuntu-iso, and --rescue-iso are required" >&2
  exit 1
fi

[[ -b "$DEVICE" ]] || { echo "Device not found: $DEVICE" >&2; exit 1; }
[[ -f "$UBUNTU_ISO" ]] || { echo "Ubuntu ISO not found: $UBUNTU_ISO" >&2; exit 1; }
[[ -f "$RESCUE_ISO" ]] || { echo "SystemRescue ISO not found: $RESCUE_ISO" >&2; exit 1; }

ROOT_DISK="/dev/$(lsblk -no pkname "$(findmnt -no SOURCE /)")"
if [[ "$DEVICE" == "$ROOT_DISK" ]]; then
  echo "Refusing to operate on root disk: $DEVICE" >&2
  exit 1
fi

if [[ "$DEVICE" =~ [0-9]$ ]]; then
  P1="${DEVICE}p1"
  P2="${DEVICE}p2"
  P3="${DEVICE}p3"
else
  P1="${DEVICE}1"
  P2="${DEVICE}2"
  P3="${DEVICE}3"
fi

echo "Preparing USB on $DEVICE"
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINTS "$DEVICE"

# Unmount any mounted partitions for target device
while read -r mnt; do
  [[ -n "$mnt" ]] && umount -R "$mnt" || true
done < <(lsblk -nrpo MOUNTPOINTS "$DEVICE" | tr ',' '\n' | awk 'NF')

wipefs -a "$DEVICE"
sgdisk --zap-all "$DEVICE"
parted -s "$DEVICE" mklabel gpt
parted -s "$DEVICE" mkpart VENTOY_EFI fat32 1MiB 2049MiB
parted -s "$DEVICE" set 1 esp on
parted -s "$DEVICE" mkpart LIVE_MULTI ext4 2049MiB "$((LIVE_SIZE_GB + 2))GiB"
parted -s "$DEVICE" mkpart EIDOS_VAULT "$((LIVE_SIZE_GB + 2))GiB" 100%
partprobe "$DEVICE"
sleep 2

mkfs.vfat -F32 -n VENTOY_EFI "$P1"
mkfs.exfat -n LIVE_MULTI "$P2"

# Encrypt and format vault partition
if [[ -n "$VAULT_PASS_FILE" ]]; then
  cryptsetup luksFormat "$P3" --type luks2 --batch-mode --key-file "$VAULT_PASS_FILE"
  cryptsetup open "$P3" eidos_vault --key-file "$VAULT_PASS_FILE"
else
  cryptsetup luksFormat "$P3" --type luks2
  cryptsetup open "$P3" eidos_vault
fi
mkfs.ext4 -F -L EIDOS_VAULT /dev/mapper/eidos_vault
sync
cryptsetup close eidos_vault || true

TMP_MNT="$(mktemp -d)"
mkdir -p "$TMP_MNT/efi" "$TMP_MNT/live"
mount "$P1" "$TMP_MNT/efi"
mount "$P2" "$TMP_MNT/live"

cp -v --no-preserve=ownership,mode,timestamps "$UBUNTU_ISO" "$TMP_MNT/live/ubuntu-live.iso"
cp -v --no-preserve=ownership,mode,timestamps "$RESCUE_ISO" "$TMP_MNT/live/systemrescue.iso"

# Create Ubuntu persistence file
truncate -s 32G "$TMP_MNT/live/casper-rw"
mkfs.ext4 -F "$TMP_MNT/live/casper-rw"

# Install removable EFI GRUB to USB
mkdir -p "$TMP_MNT/efi/boot"
grub-install \
  --target=x86_64-efi \
  --efi-directory="$TMP_MNT/efi" \
  --boot-directory="$TMP_MNT/efi/boot" \
  --removable \
  --recheck \
  --no-nvram \
  "$DEVICE"

mkdir -p "$TMP_MNT/efi/boot/grub"
cat > "$TMP_MNT/efi/boot/grub/grub.cfg" <<'EOF'
set timeout=10
set default=0

menuentry "Ubuntu Live (persistent)" {
  insmod part_gpt
  insmod fat
  insmod exfat
  search --label LIVE_MULTI --set=live
  set isofile="/ubuntu-live.iso"
  loopback loop ($live)$isofile
  linux (loop)/casper/vmlinuz boot=casper iso-scan/filename=$isofile persistent noprompt noeject ---
  initrd (loop)/casper/initrd
}

menuentry "SystemRescue Live" {
  insmod part_gpt
  insmod fat
  insmod exfat
  search --label LIVE_MULTI --set=live
  set isofile="/systemrescue.iso"
  loopback loop ($live)$isofile
  linux (loop)/sysresccd/boot/x86_64/vmlinuz img_dev=/dev/disk/by-label/LIVE_MULTI img_loop=$isofile archisobasedir=sysresccd copytoram
  initrd (loop)/sysresccd/boot/x86_64/sysresccd.img
}
EOF

sync
umount "$TMP_MNT/live"
umount "$TMP_MNT/efi"
rmdir "$TMP_MNT/live" "$TMP_MNT/efi"
rmdir "$TMP_MNT"

echo "USB build complete on $DEVICE"
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINTS "$DEVICE"
