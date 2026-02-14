#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bootstrap_eidos_machine.sh [--vault-root <path>] [--forge-root <path>] [--repo-url <url>] [--branch <name>] [--profile-dir <path>] [--setup-sync yes|no] [--install-codex yes|no] [--enable-post-boot-check yes|no]

Description:
  Rehydrates a fresh machine into an Eidos-compatible workstation:
  - installs baseline packages
  - clones or updates eidosian_forge
  - prepares eidosian_venv
  - restores host profile (configs/customizations/memory state)
  - wires MCP + Codex + optional sync mesh
EOF
}

VAULT_ROOT="/mnt/eidos_vault/eidos_clone"
FORGE_ROOT="${HOME}/eidosian_forge"
REPO_URL="https://github.com/Ace1928/eidosian_forge.git"
BRANCH="main"
PROFILE_DIR=""
SETUP_SYNC="yes"
INSTALL_CODEX="yes"
ENABLE_POST_BOOT_CHECK="yes"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vault-root) VAULT_ROOT="${2:-}"; shift 2 ;;
    --forge-root) FORGE_ROOT="${2:-}"; shift 2 ;;
    --repo-url) REPO_URL="${2:-}"; shift 2 ;;
    --branch) BRANCH="${2:-}"; shift 2 ;;
    --profile-dir) PROFILE_DIR="${2:-}"; shift 2 ;;
    --setup-sync) SETUP_SYNC="${2:-}"; shift 2 ;;
    --install-codex) INSTALL_CODEX="${2:-}"; shift 2 ;;
    --enable-post-boot-check) ENABLE_POST_BOOT_CHECK="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

as_root() {
  if [[ $EUID -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

as_root apt-get update
as_root apt-get install -y \
  git curl rsync jq ripgrep fd-find \
  python3 python3-venv python3-pip python3-dev build-essential \
  syncthing tailscale age exfatprogs cryptsetup

if [[ "$INSTALL_CODEX" == "yes" && ! -x "$(command -v codex || true)" ]]; then
  export NVM_DIR="${HOME}/.nvm"
  if [[ ! -s "${NVM_DIR}/nvm.sh" ]]; then
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  fi
  # shellcheck disable=SC1090
  source "${NVM_DIR}/nvm.sh"
  nvm install 22
  nvm alias default 22
  npm install -g @openai/codex
fi

if [[ -d "$FORGE_ROOT/.git" ]]; then
  git -C "$FORGE_ROOT" fetch origin
  git -C "$FORGE_ROOT" checkout "$BRANCH"
  git -C "$FORGE_ROOT" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$FORGE_ROOT"
fi

python3 -m venv "$FORGE_ROOT/eidosian_venv"
# shellcheck disable=SC1091
source "$FORGE_ROOT/eidosian_venv/bin/activate"
python -m pip install --upgrade pip wheel
if [[ -f "$FORGE_ROOT/archive_forge/manifests/eidosian_venv_freeze_latest.txt" ]]; then
  pip install -r "$FORGE_ROOT/archive_forge/manifests/eidosian_venv_freeze_latest.txt"
elif [[ -f "$FORGE_ROOT/requirements.txt" ]]; then
  pip install -r "$FORGE_ROOT/requirements.txt"
fi
if [[ -f "$FORGE_ROOT/eidos_mcp/pyproject.toml" ]]; then
  pip install -e "$FORGE_ROOT/eidos_mcp"
fi

if [[ -z "$PROFILE_DIR" && -d "$VAULT_ROOT/clone_kits" ]]; then
  PROFILE_DIR="$(ls -1dt "$VAULT_ROOT"/clone_kits/host_profile_* 2>/dev/null | head -n1 || true)"
fi

if [[ -n "$PROFILE_DIR" && -d "$PROFILE_DIR" ]]; then
  "$FORGE_ROOT/scripts/machine_clone/restore_new_host.sh" \
    --profile "$PROFILE_DIR" \
    --mode hybrid \
    --forge-root "$FORGE_ROOT" \
    --restore-memory yes \
    --setup-eidos-mcp yes \
    --setup-codex yes
fi

if [[ "$SETUP_SYNC" == "yes" ]]; then
  "$FORGE_ROOT/scripts/machine_clone/sync_mesh_enroll.sh" \
    --node-name "$(hostname -s)" \
    --role laptop \
    --forge-root "$FORGE_ROOT"
fi

if [[ "$ENABLE_POST_BOOT_CHECK" == "yes" ]]; then
  mkdir -p "$HOME/.config/systemd/user"
  cat > "$HOME/.config/systemd/user/eidos-post-boot.service" <<EOF
[Unit]
Description=Eidos post-boot onboarding checks
After=default.target network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=${FORGE_ROOT}/scripts/machine_clone/post_boot_onboarding.sh --forge-root ${FORGE_ROOT}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable eidos-post-boot.service || true
fi

echo "Bootstrap complete."
echo "Next actions:"
echo "1) sudo tailscale up --hostname=$(hostname -s) --accept-routes=false"
echo "2) codex --version"
echo "3) systemctl --user status eidos-mcp.service"
echo "4) reboot (post-boot checks will run via eidos-post-boot.service)"
