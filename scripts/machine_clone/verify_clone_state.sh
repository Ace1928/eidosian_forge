#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  verify_clone_state.sh --profile <host_profile_dir>

Description:
  Performs a practical parity check between current host and exported profile.
EOF
}

PROFILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$PROFILE" ]] || { echo "--profile is required" >&2; exit 1; }
[[ -d "$PROFILE" ]] || { echo "Profile dir not found: $PROFILE" >&2; exit 1; }

PACKAGES_DIR="$PROFILE/packages"
CONFIGS_DIR="$PROFILE/configs"

echo "== Package delta check =="
if [[ -f "$PACKAGES_DIR/dpkg-selections.txt" ]]; then
  src_count="$(wc -l < "$PACKAGES_DIR/dpkg-selections.txt")"
  cur_count="$(dpkg --get-selections | wc -l)"
  echo "dpkg count profile/current: $src_count / $cur_count"
fi

if [[ -f "$PACKAGES_DIR/snap-list.txt" ]] && command -v snap >/dev/null 2>&1; then
  src_snaps="$(awk 'NR>1 {print $1}' "$PACKAGES_DIR/snap-list.txt" | sort)"
  cur_snaps="$(snap list | awk 'NR>1 {print $1}' | sort)"
  echo "missing snaps:"
  comm -23 <(printf '%s\n' "$src_snaps") <(printf '%s\n' "$cur_snaps") || true
fi

echo
echo "== GNOME key settings =="
if [[ -f "$CONFIGS_DIR/gnome_key_settings.txt" ]]; then
  cat "$CONFIGS_DIR/gnome_key_settings.txt"
  echo "--- current ---"
  echo "gtk-theme=$(gsettings get org.gnome.desktop.interface gtk-theme 2>/dev/null || true)"
  echo "icon-theme=$(gsettings get org.gnome.desktop.interface icon-theme 2>/dev/null || true)"
  echo "wm-theme=$(gsettings get org.gnome.desktop.wm.preferences theme 2>/dev/null || true)"
  echo "background=$(gsettings get org.gnome.desktop.background picture-uri 2>/dev/null || true)"
  echo "font-name=$(gsettings get org.gnome.desktop.interface font-name 2>/dev/null || true)"
  echo "mono-font=$(gsettings get org.gnome.desktop.interface monospace-font-name 2>/dev/null || true)"
fi

echo
echo "== Sync service status =="
systemctl is-enabled ssh 2>/dev/null || true
systemctl is-enabled tailscaled 2>/dev/null || true
systemctl --user is-enabled syncthing.service 2>/dev/null || true

echo "Verification complete."
