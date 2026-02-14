#!/usr/bin/env bash
set -euo pipefail

LAPTOP_ID="OISJJVQ-2IQO5IL-TND6CX6-3PEZGW3-2TTHWCU-CFX7SFP-SJYSWVE-KHII3AQ"
LAPTOP_ADDR_1="tcp://100.87.208.90:22000"
LAPTOP_ADDR_2="tcp://192.168.4.165:22000"

find_syncthing_config() {
  local candidates=(
    "$HOME/.local/state/syncthing/config.xml"
    "$HOME/.config/syncthing/config.xml"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      printf '%s\n' "$c"
      return 0
    fi
  done
  return 1
}

CFG_XML="$(find_syncthing_config || true)"
if [[ -z "$CFG_XML" ]]; then
  echo "Could not find syncthing config.xml under ~/.local/state or ~/.config" >&2
  exit 1
fi

API_KEY="$(sed -n 's:.*<apikey>\(.*\)</apikey>.*:\1:p' "$CFG_XML")"
if [[ -z "$API_KEY" ]]; then
  echo "Could not parse Syncthing API key from $CFG_XML" >&2
  exit 1
fi

BASE_URL="http://127.0.0.1:8384"
if ! curl -fsS -H "X-API-Key: $API_KEY" "$BASE_URL/rest/noauth/health" >/dev/null 2>&1; then
  if ! curl -fsS -H "X-API-Key: $API_KEY" "$BASE_URL/rest/system/ping" >/dev/null 2>&1; then
    echo "Syncthing API is not reachable at $BASE_URL" >&2
    exit 1
  fi
fi

CFG_JSON="$(mktemp)"
NEW_CFG_JSON="$(mktemp)"
trap 'rm -f "$CFG_JSON" "$NEW_CFG_JSON"' EXIT

curl -fsS -H "X-API-Key: $API_KEY" "$BASE_URL/rest/config" > "$CFG_JSON"
DESKTOP_ID="$(curl -fsS -H "X-API-Key: $API_KEY" "$BASE_URL/rest/system/status" | jq -r '.myID')"

mkdir -p "$HOME/.eidosian" "$HOME/.ruff_cache" "$HOME/.benchmarks" "$HOME/.gnupg" "$HOME/.config" "$HOME/.codex"
chmod 700 "$HOME/.gnupg" || true

jq \
  --arg desktop_id "$DESKTOP_ID" \
  --arg laptop_id "$LAPTOP_ID" \
  --arg laptop_addr_1 "$LAPTOP_ADDR_1" \
  --arg laptop_addr_2 "$LAPTOP_ADDR_2" \
  '
  def ensure_device($d):
    if (.devices | map(.deviceID) | index($d.deviceID)) == null
    then .devices += [$d]
    else .devices |= map(if .deviceID == $d.deviceID then (. * $d) else . end)
    end;

  def ensure_folder($f):
    if (.folders | map(.id) | index($f.id)) == null
    then .folders += [$f]
    else .folders |= map(
      if .id == $f.id then
        . * {
          label: $f.label,
          path: $f.path,
          type: $f.type,
          rescanIntervalS: $f.rescanIntervalS,
          fsWatcherEnabled: $f.fsWatcherEnabled,
          fsWatcherDelayS: $f.fsWatcherDelayS,
          ignorePerms: $f.ignorePerms,
          ignoreDelete: $f.ignoreDelete,
          autoNormalize: $f.autoNormalize,
          devices: $f.devices
        }
      else . end)
    end;

  . as $cfg
  | ($cfg.defaults.device + {
      deviceID: $laptop_id,
      name: "Eidos-Laptop",
      addresses: ["dynamic", $laptop_addr_1, $laptop_addr_2],
      compression: "metadata",
      introducer: false,
      autoAcceptFolders: true,
      paused: false
    }) as $laptopDevice
  | ($cfg.defaults.folder + {
      id: "dot-eidosian",
      label: ".eidosian",
      path: ("/home/" + (env.USER // "lloyd") + "/.eidosian"),
      type: "receiveonly",
      rescanIntervalS: 120,
      fsWatcherEnabled: true,
      fsWatcherDelayS: 5,
      ignorePerms: false,
      ignoreDelete: false,
      autoNormalize: true,
      paused: false,
      devices: [
        {deviceID: $desktop_id, introducedBy: "", encryptionPassword: ""},
        {deviceID: $laptop_id, introducedBy: "", encryptionPassword: ""}
      ]
    }) as $f1
  | ($cfg.defaults.folder + {
      id: "dot-ruff-cache",
      label: ".ruff_cache",
      path: ("/home/" + (env.USER // "lloyd") + "/.ruff_cache"),
      type: "receiveonly",
      rescanIntervalS: 120,
      fsWatcherEnabled: true,
      fsWatcherDelayS: 5,
      ignorePerms: false,
      ignoreDelete: false,
      autoNormalize: true,
      paused: false,
      devices: [
        {deviceID: $desktop_id, introducedBy: "", encryptionPassword: ""},
        {deviceID: $laptop_id, introducedBy: "", encryptionPassword: ""}
      ]
    }) as $f2
  | ($cfg.defaults.folder + {
      id: "dot-benchmarks",
      label: ".benchmarks",
      path: ("/home/" + (env.USER // "lloyd") + "/.benchmarks"),
      type: "receiveonly",
      rescanIntervalS: 120,
      fsWatcherEnabled: true,
      fsWatcherDelayS: 5,
      ignorePerms: false,
      ignoreDelete: false,
      autoNormalize: true,
      paused: false,
      devices: [
        {deviceID: $desktop_id, introducedBy: "", encryptionPassword: ""},
        {deviceID: $laptop_id, introducedBy: "", encryptionPassword: ""}
      ]
    }) as $f3
  | ($cfg.defaults.folder + {
      id: "dot-gnupg",
      label: ".gnupg",
      path: ("/home/" + (env.USER // "lloyd") + "/.gnupg"),
      type: "receiveonly",
      rescanIntervalS: 120,
      fsWatcherEnabled: true,
      fsWatcherDelayS: 5,
      ignorePerms: false,
      ignoreDelete: false,
      autoNormalize: true,
      paused: false,
      devices: [
        {deviceID: $desktop_id, introducedBy: "", encryptionPassword: ""},
        {deviceID: $laptop_id, introducedBy: "", encryptionPassword: ""}
      ]
    }) as $f4
  | ($cfg.defaults.folder + {
      id: "dot-config",
      label: ".config",
      path: ("/home/" + (env.USER // "lloyd") + "/.config"),
      type: "receiveonly",
      rescanIntervalS: 120,
      fsWatcherEnabled: true,
      fsWatcherDelayS: 5,
      ignorePerms: false,
      ignoreDelete: false,
      autoNormalize: true,
      paused: false,
      devices: [
        {deviceID: $desktop_id, introducedBy: "", encryptionPassword: ""},
        {deviceID: $laptop_id, introducedBy: "", encryptionPassword: ""}
      ]
    }) as $f5
  | ($cfg.defaults.folder + {
      id: "dot-codex",
      label: ".codex",
      path: ("/home/" + (env.USER // "lloyd") + "/.codex"),
      type: "receiveonly",
      rescanIntervalS: 120,
      fsWatcherEnabled: true,
      fsWatcherDelayS: 5,
      ignorePerms: false,
      ignoreDelete: false,
      autoNormalize: true,
      paused: false,
      devices: [
        {deviceID: $desktop_id, introducedBy: "", encryptionPassword: ""},
        {deviceID: $laptop_id, introducedBy: "", encryptionPassword: ""}
      ]
    }) as $f6
  | ensure_device($laptopDevice)
  | ensure_folder($f1)
  | ensure_folder($f2)
  | ensure_folder($f3)
  | ensure_folder($f4)
  | ensure_folder($f5)
  | ensure_folder($f6)
  ' "$CFG_JSON" > "$NEW_CFG_JSON"

curl -fsS -X PUT \
  -H "X-API-Key: $API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @"$NEW_CFG_JSON" \
  "$BASE_URL/rest/config" >/dev/null

curl -fsS -X POST -H "X-API-Key: $API_KEY" "$BASE_URL/rest/system/restart" >/dev/null || true
sleep 5

API_KEY="$(sed -n 's:.*<apikey>\(.*\)</apikey>.*:\1:p' "$CFG_XML")"

for fid in dot-eidosian dot-ruff-cache dot-benchmarks dot-gnupg dot-config dot-codex; do
  curl -fsS -X POST -H "X-API-Key: $API_KEY" "$BASE_URL/rest/db/scan?folder=${fid}" >/dev/null || true
done

echo "Desktop Syncthing configured for Eidos laptop hidden-folder sync."
echo "Watch progress in Syncthing UI or run:"
echo "curl -H 'X-API-Key: <apikey>' 'http://127.0.0.1:8384/rest/db/completion?folder=dot-config&device=${LAPTOP_ID}'"
