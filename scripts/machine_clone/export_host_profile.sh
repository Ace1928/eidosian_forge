#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  export_host_profile.sh --output <dir> [--include-secrets yes|no] [--secret-passphrase-file <file>] [--age-recipient <recipient>]

Description:
  Exports a reproducible host profile bundle with package manifests, system state,
  user config/customization assets, and optionally encrypted secrets.
EOF
}

OUTPUT_DIR=""
INCLUDE_SECRETS="no"
PASSPHRASE_FILE=""
AGE_RECIPIENT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT_DIR="${2:-}"; shift 2 ;;
    --include-secrets)
      INCLUDE_SECRETS="${2:-}"; shift 2 ;;
    --secret-passphrase-file)
      PASSPHRASE_FILE="${2:-}"; shift 2 ;;
    --age-recipient)
      AGE_RECIPIENT="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "--output is required" >&2
  exit 1
fi

if [[ "$INCLUDE_SECRETS" != "yes" && "$INCLUDE_SECRETS" != "no" ]]; then
  echo "--include-secrets must be yes or no" >&2
  exit 1
fi

STAMP="$(date +%Y-%m-%d_%H%M%S)"
HOSTNAME_SAFE="$(hostname | tr -cd '[:alnum:]_.-')"
PROFILE_ROOT="${OUTPUT_DIR%/}/host_profile_${HOSTNAME_SAFE}_${STAMP}"

mkdir -p \
  "$PROFILE_ROOT/system" \
  "$PROFILE_ROOT/packages" \
  "$PROFILE_ROOT/configs" \
  "$PROFILE_ROOT/assets" \
  "$PROFILE_ROOT/secrets" \
  "$PROFILE_ROOT/manifests"

echo "Exporting host profile to: $PROFILE_ROOT"

# Core system identity
cp /etc/os-release "$PROFILE_ROOT/system/os-release"
uname -a > "$PROFILE_ROOT/system/uname.txt"
lsblk -o NAME,SIZE,FSTYPE,LABEL,UUID,MOUNTPOINTS > "$PROFILE_ROOT/system/lsblk.txt"
blkid > "$PROFILE_ROOT/system/blkid.txt" || true
cat /etc/fstab > "$PROFILE_ROOT/system/fstab.txt"
df -hT > "$PROFILE_ROOT/system/df-hT.txt"
timedatectl > "$PROFILE_ROOT/system/timedatectl.txt" || true
locale > "$PROFILE_ROOT/system/locale.txt" || true

# Package/runtime manifests
dpkg --get-selections > "$PROFILE_ROOT/packages/dpkg-selections.txt"
apt-mark showmanual > "$PROFILE_ROOT/packages/apt-manual.txt" || true
if command -v snap >/dev/null 2>&1; then
  snap list > "$PROFILE_ROOT/packages/snap-list.txt" || true
fi
python3 -m pip freeze > "$PROFILE_ROOT/packages/pip-freeze.txt" || true
if command -v pipx >/dev/null 2>&1; then
  pipx list > "$PROFILE_ROOT/packages/pipx-list.txt" || true
fi
if command -v ollama >/dev/null 2>&1; then
  ollama list > "$PROFILE_ROOT/packages/ollama-models.txt" || true
fi
if command -v npm >/dev/null 2>&1; then
  npm list -g --depth=0 > "$PROFILE_ROOT/packages/npm-global.txt" || true
fi
if command -v codex >/dev/null 2>&1; then
  codex --version > "$PROFILE_ROOT/packages/codex-version.txt" || true
fi

# Desktop customization capture
dconf dump / > "$PROFILE_ROOT/configs/dconf_dump.ini" || true
{
  echo "gtk-theme=$(gsettings get org.gnome.desktop.interface gtk-theme 2>/dev/null || true)"
  echo "icon-theme=$(gsettings get org.gnome.desktop.interface icon-theme 2>/dev/null || true)"
  echo "wm-theme=$(gsettings get org.gnome.desktop.wm.preferences theme 2>/dev/null || true)"
  echo "background=$(gsettings get org.gnome.desktop.background picture-uri 2>/dev/null || true)"
  echo "font-name=$(gsettings get org.gnome.desktop.interface font-name 2>/dev/null || true)"
  echo "mono-font=$(gsettings get org.gnome.desktop.interface monospace-font-name 2>/dev/null || true)"
} > "$PROFILE_ROOT/configs/gnome_key_settings.txt"

mkdir -p "$PROFILE_ROOT/configs/home"
for f in \
  "$HOME/.bashrc" \
  "$HOME/.profile" \
  "$HOME/.gitconfig" \
  "$HOME/.cursorignore"; do
  [[ -f "$f" ]] && cp -a "$f" "$PROFILE_ROOT/configs/home/"
done

for d in \
  "$HOME/.codex" \
  "$HOME/.gemini"; do
  if [[ -d "$d" ]]; then
    mkdir -p "$PROFILE_ROOT/configs/home/$(basename "$d")"
    rsync -a \
      --exclude 'auth.json' \
      --exclude 'oauth_creds.json' \
      --exclude 'history.jsonl' \
      --exclude 'sessions/' \
      --exclude 'tmp/' \
      "$d/" "$PROFILE_ROOT/configs/home/$(basename "$d")/"
  fi
done

if [[ -d "$HOME/.config" ]]; then
  rsync -a \
    --exclude '*/Cache/' \
    --exclude '*/cache/' \
    --exclude '*/GPUCache/' \
    --exclude '*.log' \
    "$HOME/.config/" "$PROFILE_ROOT/configs/home/.config/"
fi

if [[ -d "$HOME/.local/share" ]]; then
  mkdir -p "$PROFILE_ROOT/configs/home/.local/share"
  for d in fonts themes icons applications gnome-shell backgrounds; do
    [[ -d "$HOME/.local/share/$d" ]] && rsync -a "$HOME/.local/share/$d/" "$PROFILE_ROOT/configs/home/.local/share/$d/"
  done
fi

# Assets
if [[ -d "/usr/share/backgrounds" ]]; then
  rsync -a "/usr/share/backgrounds/" "$PROFILE_ROOT/assets/usr-share-backgrounds/"
fi
if [[ -d "$HOME/Pictures" ]]; then
  rsync -a "$HOME/Pictures/" "$PROFILE_ROOT/assets/home-pictures/"
fi
if [[ -d "$HOME/.local/share/fonts" ]]; then
  rsync -a "$HOME/.local/share/fonts/" "$PROFILE_ROOT/assets/home-fonts/"
fi
if [[ -d "$HOME/.fonts" ]]; then
  rsync -a "$HOME/.fonts/" "$PROFILE_ROOT/assets/home-dot-fonts/"
fi
fc-list > "$PROFILE_ROOT/assets/font-catalog.txt" || true

# Eidosian continuity state capture
FORGE_ROOT="/home/lloyd/eidosian_forge"
if [[ -d "$FORGE_ROOT" ]]; then
  mkdir -p "$PROFILE_ROOT/assets/eidosian_forge_state"
  for p in \
    "$FORGE_ROOT/memory_data.json" \
    "$FORGE_ROOT/data/tiered_memory" \
    "$FORGE_ROOT/memory" \
    "$FORGE_ROOT/archive_forge/manifests"; do
    if [[ -e "$p" ]]; then
      rsync -a "$p" "$PROFILE_ROOT/assets/eidosian_forge_state/"
    fi
  done
  git -C "$FORGE_ROOT" rev-parse HEAD > "$PROFILE_ROOT/assets/eidosian_forge_state/forge_git_head.txt" || true
fi

# Optional encrypted secrets pack
if [[ "$INCLUDE_SECRETS" == "yes" ]]; then
  TMP_SECRETS_DIR="$PROFILE_ROOT/secrets/plain"
  mkdir -p "$TMP_SECRETS_DIR"
  for p in \
    "$HOME/.ssh" \
    "$HOME/.gnupg" \
    "$HOME/.password-store"; do
    if [[ -e "$p" ]]; then
      rsync -a --ignore-errors "$p" "$TMP_SECRETS_DIR/" || true
    fi
  done

  tar -C "$TMP_SECRETS_DIR" -cf "$PROFILE_ROOT/secrets/secrets.tar" .
  rm -rf "$TMP_SECRETS_DIR"

  if [[ -n "$AGE_RECIPIENT" ]]; then
    if ! command -v age >/dev/null 2>&1; then
      echo "age not found; cannot encrypt with --age-recipient" >&2
      exit 1
    fi
    age -r "$AGE_RECIPIENT" -o "$PROFILE_ROOT/secrets/secrets.tar.age" "$PROFILE_ROOT/secrets/secrets.tar"
    rm -f "$PROFILE_ROOT/secrets/secrets.tar"
  elif [[ -n "$PASSPHRASE_FILE" ]]; then
    openssl enc -aes-256-cbc -pbkdf2 -salt \
      -in "$PROFILE_ROOT/secrets/secrets.tar" \
      -out "$PROFILE_ROOT/secrets/secrets.tar.enc" \
      -pass "file:${PASSPHRASE_FILE}"
    rm -f "$PROFILE_ROOT/secrets/secrets.tar"
  else
    echo "Secrets exported unencrypted (no --secret-passphrase-file or --age-recipient provided)." >&2
  fi
fi

DPKG_COUNT="$(wc -l < "$PROFILE_ROOT/packages/dpkg-selections.txt" || echo 0)"
SNAP_COUNT="$( [[ -f "$PROFILE_ROOT/packages/snap-list.txt" ]] && tail -n +2 "$PROFILE_ROOT/packages/snap-list.txt" | wc -l || echo 0 )"
FILE_COUNT="$(find "$PROFILE_ROOT" -type f | wc -l)"

cat > "$PROFILE_ROOT/manifests/host_profile_manifest.v1.json" <<EOF
{
  "schema_version": "v1",
  "created_at": "${STAMP}",
  "hostname": "${HOSTNAME_SAFE}",
  "include_secrets": "${INCLUDE_SECRETS}",
  "dpkg_package_count": ${DPKG_COUNT},
  "snap_package_count": ${SNAP_COUNT},
  "bundle_file_count": ${FILE_COUNT},
  "paths": {
    "system": "system",
    "packages": "packages",
    "configs": "configs",
    "assets": "assets",
    "secrets": "secrets"
  }
}
EOF

echo "$PROFILE_ROOT"
