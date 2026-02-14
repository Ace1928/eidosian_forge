#!/usr/bin/env bash
set -euo pipefail

LAPTOP_ID="OISJJVQ-2IQO5IL-TND6CX6-3PEZGW3-2TTHWCU-CFX7SFP-SJYSWVE-KHII3AQ"
LAPTOP_ADDR_1="tcp://100.87.208.90:22000"
LAPTOP_ADDR_2="tcp://192.168.4.165:22000"

resolve_profile_script() {
  local candidates=(
    "$HOME/eidosian_forge/scripts/syncthing_apply_parity_profile.py"
    "$HOME/.eidosian/agent_bridge/bin/syncthing_apply_parity_profile.py"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf "%s\n" "$candidate"
      return 0
    fi
  done
  return 1
}

PROFILE_SCRIPT="$(resolve_profile_script || true)"
if [[ -z "$PROFILE_SCRIPT" ]]; then
  echo "Could not find syncthing_apply_parity_profile.py." >&2
  echo "Expected one of:" >&2
  echo "  ~/eidosian_forge/scripts/syncthing_apply_parity_profile.py" >&2
  echo "  ~/.eidosian/agent_bridge/bin/syncthing_apply_parity_profile.py" >&2
  exit 1
fi

python "$PROFILE_SCRIPT" \
  --remote-id "$LAPTOP_ID" \
  --remote-name "Eidos-Laptop" \
  --remote-addr "$LAPTOP_ADDR_1" \
  --remote-addr "$LAPTOP_ADDR_2"

echo "Desktop full parity profile applied."
