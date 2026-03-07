#!/usr/bin/env bash
set -euo pipefail

want_commands=(starship uv eza bat tmux pulseaudio)
termux_packages=(starship uv eza bat tmux pulseaudio termux-api)
apt_packages=(starship uv eza bat tmux pulseaudio)
dnf_packages=(starship uv eza bat tmux pulseaudio)
pacman_packages=(starship uv eza bat tmux pulseaudio)

missing=()
for cmd in "${want_commands[@]}"; do
    command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
done

if [ "${#missing[@]}" -eq 0 ]; then
    printf 'all shell prerequisites are already installed\n'
    exit 0
fi

if command -v pkg >/dev/null 2>&1 && [ -n "${PREFIX:-}" ] && printf '%s' "${PREFIX}" | grep -q 'com.termux'; then
    pkg install -y "${termux_packages[@]}"
elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y "${apt_packages[@]}"
elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y "${dnf_packages[@]}"
elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -Sy --needed --noconfirm "${pacman_packages[@]}"
else
    printf 'missing commands: %s\n' "${missing[*]}" >&2
    exit 1
fi
