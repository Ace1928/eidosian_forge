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
    [--live-fs ext4|exfat] \
    [--casper-rw-size-gb 32] \
    [--post-verify yes|no] \
    [--vault-passphrase-file /path/passphrase.txt]

Utility mode:
  build_dual_live_usb.sh --print-grub-config [--live-fs ext4|exfat]

Description:
  Creates a GPT layout with:
  - partition 1: FAT32 EFI boot (2GiB)
  - partition 2: LIVE_MULTI (default ext4 for Ubuntu casper compatibility)
  - partition 3: LUKS2 + ext4 EIDOS_VAULT

  Installs GRUB EFI to the USB and writes loopback boot entries for both ISOs.
EOF
}

DEVICE=""
UBUNTU_ISO=""
RESCUE_ISO=""
LIVE_SIZE_GB="126"
LIVE_FS="ext4"
CASPER_RW_SIZE_GB="32"
VAULT_PASS_FILE=""
PRINT_GRUB_CONFIG="no"
POST_VERIFY="yes"

TMP_MNT=""
P1=""
P2=""
P3=""
MOUNTED_EFI="no"
MOUNTED_LIVE="no"
VAULT_OPEN="no"

cleanup() {
  set +e
  if [[ "$MOUNTED_LIVE" == "yes" && -n "$TMP_MNT" ]]; then
    umount "$TMP_MNT/live" >/dev/null 2>&1 || true
  fi
  if [[ "$MOUNTED_EFI" == "yes" && -n "$TMP_MNT" ]]; then
    umount "$TMP_MNT/efi" >/dev/null 2>&1 || true
  fi
  if [[ "$VAULT_OPEN" == "yes" ]]; then
    cryptsetup close eidos_vault >/dev/null 2>&1 || true
  fi
  if [[ -n "$TMP_MNT" && -d "$TMP_MNT" ]]; then
    rmdir "$TMP_MNT/live" "$TMP_MNT/efi" >/dev/null 2>&1 || true
    rmdir "$TMP_MNT" >/dev/null 2>&1 || true
  fi
}

live_fs_grub_module() {
  case "$LIVE_FS" in
    ext4) echo "ext2" ;;
    exfat) echo "exfat" ;;
    *) echo "Unsupported LIVE_FS: $LIVE_FS" >&2; return 1 ;;
  esac
}

validate_live_fs() {
  case "$LIVE_FS" in
    ext4|exfat) ;;
    *)
      echo "Invalid --live-fs value: $LIVE_FS (expected ext4 or exfat)" >&2
      return 1
      ;;
  esac
}

write_grub_config() {
  local out_path="$1"
  local live_module
  live_module="$(live_fs_grub_module)"
  cat > "$out_path" <<EOF
set timeout=10
set default=0

function set_iso_path {
  if [ -e (\$live)/\$1 ]; then
    set isofile="/\$1"
  elif [ -e (\$live)/live/\$1 ]; then
    set isofile="/live/\$1"
  else
    echo "ISO not found: \$1"
    sleep 5
  fi
}

menuentry "Ubuntu Live (persistent)" {
  insmod part_gpt
  insmod fat
  insmod ${live_module}
  search --no-floppy --label LIVE_MULTI --set=live
  set_iso_path ubuntu-live.iso
  loopback loop (\$live)\$isofile
  linux (loop)/casper/vmlinuz boot=casper iso-scan/filename=\$isofile findiso=\$isofile persistent noprompt noeject ---
  initrd (loop)/casper/initrd
}

menuentry "SystemRescue Live" {
  insmod part_gpt
  insmod fat
  insmod ${live_module}
  search --no-floppy --label LIVE_MULTI --set=live
  set_iso_path systemrescue.iso
  loopback loop (\$live)\$isofile
  linux (loop)/sysresccd/boot/x86_64/vmlinuz img_dev=/dev/disk/by-label/LIVE_MULTI img_loop=\$isofile archisobasedir=sysresccd copytoram
  initrd (loop)/sysresccd/boot/x86_64/sysresccd.img
}
EOF
}

validate_numeric() {
  local value="$1"
  local name="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "$name must be an integer, got: $value" >&2
    return 1
  }
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --device) DEVICE="${2:-}"; shift 2 ;;
      --ubuntu-iso) UBUNTU_ISO="${2:-}"; shift 2 ;;
      --rescue-iso) RESCUE_ISO="${2:-}"; shift 2 ;;
      --live-size-gb) LIVE_SIZE_GB="${2:-}"; shift 2 ;;
      --live-fs) LIVE_FS="${2:-}"; shift 2 ;;
      --casper-rw-size-gb) CASPER_RW_SIZE_GB="${2:-}"; shift 2 ;;
      --post-verify) POST_VERIFY="${2:-}"; shift 2 ;;
      --vault-passphrase-file) VAULT_PASS_FILE="${2:-}"; shift 2 ;;
      --print-grub-config) PRINT_GRUB_CONFIG="yes"; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
    esac
  done
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing command: $1" >&2
    exit 1
  }
}

validate_iso_layout() {
  local iso_path="$1"
  local expected_path="$2"
  local label="$3"
  local tmp_iso
  tmp_iso="$(mktemp -d)"
  if ! mount -o loop,ro "$iso_path" "$tmp_iso"; then
    rmdir "$tmp_iso"
    echo "Failed to mount $label ISO for validation: $iso_path" >&2
    return 1
  fi
  if [[ ! -e "$tmp_iso/$expected_path" ]]; then
    umount "$tmp_iso" || true
    rmdir "$tmp_iso" || true
    echo "$label ISO missing required path: /$expected_path" >&2
    return 1
  fi
  umount "$tmp_iso"
  rmdir "$tmp_iso"
}

setup_partition_paths() {
  if [[ "$DEVICE" =~ [0-9]$ ]]; then
    P1="${DEVICE}p1"
    P2="${DEVICE}p2"
    P3="${DEVICE}p3"
  else
    P1="${DEVICE}1"
    P2="${DEVICE}2"
    P3="${DEVICE}3"
  fi
}

format_live_partition() {
  case "$LIVE_FS" in
    ext4)
      mkfs.ext4 -F -L LIVE_MULTI "$P2"
      ;;
    exfat)
      mkfs.exfat -n LIVE_MULTI "$P2"
      ;;
    *)
      echo "Unsupported LIVE_FS: $LIVE_FS" >&2
      exit 1
      ;;
  esac
}

main() {
  parse_args "$@"
  validate_live_fs
  validate_numeric "$LIVE_SIZE_GB" "live-size-gb"
  validate_numeric "$CASPER_RW_SIZE_GB" "casper-rw-size-gb"
  [[ "$POST_VERIFY" == "yes" || "$POST_VERIFY" == "no" ]] || {
    echo "post-verify must be yes or no, got: $POST_VERIFY" >&2
    exit 1
  }

  if [[ "$PRINT_GRUB_CONFIG" == "yes" ]]; then
    tmp_cfg="$(mktemp)"
    write_grub_config "$tmp_cfg"
    cat "$tmp_cfg"
    rm -f "$tmp_cfg"
    exit 0
  fi

  if [[ $EUID -ne 0 ]]; then
    echo "Run as root." >&2
    exit 1
  fi

  for cmd in parted sgdisk wipefs mkfs.vfat mkfs.ext4 cryptsetup grub-install partprobe mount umount; do
    require_cmd "$cmd"
  done
  if [[ "$LIVE_FS" == "exfat" ]]; then
    require_cmd mkfs.exfat
    echo "Warning: exFAT LIVE partition can fail Ubuntu casper ISO discovery on some hardware." >&2
  fi

  if [[ -z "$DEVICE" || -z "$UBUNTU_ISO" || -z "$RESCUE_ISO" ]]; then
    echo "--device, --ubuntu-iso, and --rescue-iso are required" >&2
    exit 1
  fi

  [[ -b "$DEVICE" ]] || { echo "Device not found: $DEVICE" >&2; exit 1; }
  [[ -f "$UBUNTU_ISO" ]] || { echo "Ubuntu ISO not found: $UBUNTU_ISO" >&2; exit 1; }
  [[ -f "$RESCUE_ISO" ]] || { echo "SystemRescue ISO not found: $RESCUE_ISO" >&2; exit 1; }

  validate_iso_layout "$UBUNTU_ISO" "casper/vmlinuz" "Ubuntu"
  validate_iso_layout "$UBUNTU_ISO" "casper/initrd" "Ubuntu"
  validate_iso_layout "$RESCUE_ISO" "sysresccd/boot/x86_64/vmlinuz" "SystemRescue"
  validate_iso_layout "$RESCUE_ISO" "sysresccd/boot/x86_64/sysresccd.img" "SystemRescue"

  ROOT_DISK="/dev/$(lsblk -no pkname "$(findmnt -no SOURCE /)")"
  if [[ "$DEVICE" == "$ROOT_DISK" ]]; then
    echo "Refusing to operate on root disk: $DEVICE" >&2
    exit 1
  fi

  setup_partition_paths
  trap cleanup EXIT

  echo "Preparing USB on $DEVICE"
  lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINTS "$DEVICE"

  # Unmount any mounted partitions for target device.
  while read -r mnt; do
    [[ -n "$mnt" ]] && umount -R "$mnt" || true
  done < <(lsblk -nrpo MOUNTPOINTS "$DEVICE" | tr ',' '\n' | awk 'NF')

  wipefs -a "$DEVICE"
  sgdisk --zap-all "$DEVICE"
  parted -s "$DEVICE" mklabel gpt
  parted -s "$DEVICE" mkpart VENTOY_EFI fat32 1MiB 2049MiB
  parted -s "$DEVICE" set 1 esp on
  parted -s "$DEVICE" mkpart LIVE_MULTI "$LIVE_FS" 2049MiB "$((LIVE_SIZE_GB + 2))GiB"
  parted -s "$DEVICE" mkpart EIDOS_VAULT "$((LIVE_SIZE_GB + 2))GiB" 100%
  partprobe "$DEVICE"
  sleep 2

  mkfs.vfat -F32 -n VENTOY_EFI "$P1"
  format_live_partition

  # Encrypt and format vault partition.
  if [[ -n "$VAULT_PASS_FILE" ]]; then
    cryptsetup luksFormat "$P3" --type luks2 --batch-mode --key-file "$VAULT_PASS_FILE"
    cryptsetup open "$P3" eidos_vault --key-file "$VAULT_PASS_FILE"
  else
    cryptsetup luksFormat "$P3" --type luks2
    cryptsetup open "$P3" eidos_vault
  fi
  VAULT_OPEN="yes"
  mkfs.ext4 -F -L EIDOS_VAULT /dev/mapper/eidos_vault
  sync
  cryptsetup close eidos_vault
  VAULT_OPEN="no"

  TMP_MNT="$(mktemp -d)"
  mkdir -p "$TMP_MNT/efi" "$TMP_MNT/live"
  mount "$P1" "$TMP_MNT/efi"
  MOUNTED_EFI="yes"
  mount "$P2" "$TMP_MNT/live"
  MOUNTED_LIVE="yes"

  cp -v --no-preserve=ownership,mode,timestamps "$UBUNTU_ISO" "$TMP_MNT/live/ubuntu-live.iso"
  cp -v --no-preserve=ownership,mode,timestamps "$RESCUE_ISO" "$TMP_MNT/live/systemrescue.iso"

  # Create Ubuntu persistence file.
  truncate -s "${CASPER_RW_SIZE_GB}G" "$TMP_MNT/live/casper-rw"
  mkfs.ext4 -F "$TMP_MNT/live/casper-rw"

  [[ -f "$TMP_MNT/live/ubuntu-live.iso" ]] || { echo "Failed to copy ubuntu-live.iso" >&2; exit 1; }
  [[ -f "$TMP_MNT/live/systemrescue.iso" ]] || { echo "Failed to copy systemrescue.iso" >&2; exit 1; }
  [[ -f "$TMP_MNT/live/casper-rw" ]] || { echo "Failed to create casper-rw" >&2; exit 1; }

  # Install removable EFI GRUB to USB.
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
  write_grub_config "$TMP_MNT/efi/boot/grub/grub.cfg"

  if command -v grub-script-check >/dev/null 2>&1; then
    grub-script-check "$TMP_MNT/efi/boot/grub/grub.cfg"
  fi

  sync
  umount "$TMP_MNT/live"
  MOUNTED_LIVE="no"
  umount "$TMP_MNT/efi"
  MOUNTED_EFI="no"
  rmdir "$TMP_MNT/live" "$TMP_MNT/efi"
  rmdir "$TMP_MNT"
  TMP_MNT=""
  trap - EXIT

  echo "USB build complete on $DEVICE"
  lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINTS "$DEVICE"

  if [[ "$POST_VERIFY" == "yes" ]]; then
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -x "$script_dir/verify_dual_live_usb.sh" ]]; then
      "$script_dir/verify_dual_live_usb.sh" --device "$DEVICE"
    else
      echo "Post-verify skipped: verify_dual_live_usb.sh not found/executable" >&2
    fi
  fi
}

main "$@"
