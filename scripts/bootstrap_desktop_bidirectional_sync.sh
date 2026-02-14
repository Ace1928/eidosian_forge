#!/usr/bin/env bash
set -euo pipefail

LAPTOP_ID="OISJJVQ-2IQO5IL-TND6CX6-3PEZGW3-2TTHWCU-CFX7SFP-SJYSWVE-KHII3AQ"
LAPTOP_ADDR_1="tcp://100.87.208.90:22000"
LAPTOP_ADDR_2="tcp://192.168.4.165:22000"
HOME_DIR="${HOME}"

find_syncthing_config() {
  local candidates=(
    "$HOME_DIR/.local/state/syncthing/config.xml"
    "$HOME_DIR/.config/syncthing/config.xml"
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
  echo "Could not find syncthing config.xml" >&2
  exit 1
fi

API_KEY="$(sed -n 's:.*<apikey>\(.*\)</apikey>.*:\1:p' "$CFG_XML")"
if [[ -z "$API_KEY" ]]; then
  echo "Could not parse Syncthing API key from $CFG_XML" >&2
  exit 1
fi

BASE_URL="http://127.0.0.1:8384"
if ! curl -fsS -H "X-API-Key: $API_KEY" "$BASE_URL/rest/system/ping" >/dev/null 2>&1; then
  echo "Syncthing API not reachable at $BASE_URL" >&2
  exit 1
fi

mkdir -p \
  "$HOME_DIR/.eidosian" \
  "$HOME_DIR/.ruff_cache" \
  "$HOME_DIR/.benchmarks" \
  "$HOME_DIR/.gnupg" \
  "$HOME_DIR/.config" \
  "$HOME_DIR/.codex" \
  "$HOME_DIR/.local/share/backgrounds" \
  "$HOME_DIR/.local/share/wallpapers" \
  "$HOME_DIR/.local/share/plasma" \
  "$HOME_DIR/.local/share/icons"
chmod 700 "$HOME_DIR/.gnupg" || true

CFG_JSON="$(mktemp)"
NEW_CFG_JSON="$(mktemp)"
trap 'rm -f "$CFG_JSON" "$NEW_CFG_JSON"' EXIT

curl -fsS -H "X-API-Key: $API_KEY" "$BASE_URL/rest/config" > "$CFG_JSON"
DESKTOP_ID="$(curl -fsS -H "X-API-Key: $API_KEY" "$BASE_URL/rest/system/status" | jq -r '.myID')"

jq \
  --arg desktop_id "$DESKTOP_ID" \
  --arg laptop_id "$LAPTOP_ID" \
  --arg laptop_addr_1 "$LAPTOP_ADDR_1" \
  --arg laptop_addr_2 "$LAPTOP_ADDR_2" \
  --arg home_dir "$HOME_DIR" \
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
          paused: false,
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
      id: "dot-eidosian", label: ".eidosian", path: ($home_dir + "/.eidosian"), type: "sendreceive", rescanIntervalS: 120, fsWatcherEnabled: true, fsWatcherDelayS: 5, ignorePerms: false, ignoreDelete: false, autoNormalize: true, paused: false,
      devices: [{deviceID:$desktop_id,introducedBy:"",encryptionPassword:""},{deviceID:$laptop_id,introducedBy:"",encryptionPassword:""}]}) as $f1
  | ($cfg.defaults.folder + {
      id: "dot-ruff-cache", label: ".ruff_cache", path: ($home_dir + "/.ruff_cache"), type: "sendreceive", rescanIntervalS: 120, fsWatcherEnabled: true, fsWatcherDelayS: 5, ignorePerms: false, ignoreDelete: false, autoNormalize: true, paused: false,
      devices: [{deviceID:$desktop_id,introducedBy:"",encryptionPassword:""},{deviceID:$laptop_id,introducedBy:"",encryptionPassword:""}]}) as $f2
  | ($cfg.defaults.folder + {
      id: "dot-benchmarks", label: ".benchmarks", path: ($home_dir + "/.benchmarks"), type: "sendreceive", rescanIntervalS: 120, fsWatcherEnabled: true, fsWatcherDelayS: 5, ignorePerms: false, ignoreDelete: false, autoNormalize: true, paused: false,
      devices: [{deviceID:$desktop_id,introducedBy:"",encryptionPassword:""},{deviceID:$laptop_id,introducedBy:"",encryptionPassword:""}]}) as $f3
  | ($cfg.defaults.folder + {
      id: "dot-gnupg", label: ".gnupg", path: ($home_dir + "/.gnupg"), type: "sendreceive", rescanIntervalS: 120, fsWatcherEnabled: true, fsWatcherDelayS: 5, ignorePerms: false, ignoreDelete: false, autoNormalize: true, paused: false,
      devices: [{deviceID:$desktop_id,introducedBy:"",encryptionPassword:""},{deviceID:$laptop_id,introducedBy:"",encryptionPassword:""}]}) as $f4
  | ($cfg.defaults.folder + {
      id: "dot-config", label: ".config", path: ($home_dir + "/.config"), type: "sendreceive", rescanIntervalS: 120, fsWatcherEnabled: true, fsWatcherDelayS: 5, ignorePerms: false, ignoreDelete: false, autoNormalize: true, paused: false,
      devices: [{deviceID:$desktop_id,introducedBy:"",encryptionPassword:""},{deviceID:$laptop_id,introducedBy:"",encryptionPassword:""}]}) as $f5
  | ($cfg.defaults.folder + {
      id: "dot-codex", label: ".codex", path: ($home_dir + "/.codex"), type: "sendreceive", rescanIntervalS: 120, fsWatcherEnabled: true, fsWatcherDelayS: 5, ignorePerms: false, ignoreDelete: false, autoNormalize: true, paused: false,
      devices: [{deviceID:$desktop_id,introducedBy:"",encryptionPassword:""},{deviceID:$laptop_id,introducedBy:"",encryptionPassword:""}]}) as $f6
  | ($cfg.defaults.folder + {
      id: "dot-local-share-backgrounds", label: ".local/share/backgrounds", path: ($home_dir + "/.local/share/backgrounds"), type: "sendreceive", rescanIntervalS: 120, fsWatcherEnabled: true, fsWatcherDelayS: 5, ignorePerms: false, ignoreDelete: false, autoNormalize: true, paused: false,
      devices: [{deviceID:$desktop_id,introducedBy:"",encryptionPassword:""},{deviceID:$laptop_id,introducedBy:"",encryptionPassword:""}]}) as $f7
  | ($cfg.defaults.folder + {
      id: "dot-local-share-wallpapers", label: ".local/share/wallpapers", path: ($home_dir + "/.local/share/wallpapers"), type: "sendreceive", rescanIntervalS: 120, fsWatcherEnabled: true, fsWatcherDelayS: 5, ignorePerms: false, ignoreDelete: false, autoNormalize: true, paused: false,
      devices: [{deviceID:$desktop_id,introducedBy:"",encryptionPassword:""},{deviceID:$laptop_id,introducedBy:"",encryptionPassword:""}]}) as $f8
  | ($cfg.defaults.folder + {
      id: "dot-local-share-plasma", label: ".local/share/plasma", path: ($home_dir + "/.local/share/plasma"), type: "sendreceive", rescanIntervalS: 120, fsWatcherEnabled: true, fsWatcherDelayS: 5, ignorePerms: false, ignoreDelete: false, autoNormalize: true, paused: false,
      devices: [{deviceID:$desktop_id,introducedBy:"",encryptionPassword:""},{deviceID:$laptop_id,introducedBy:"",encryptionPassword:""}]}) as $f9
  | ($cfg.defaults.folder + {
      id: "dot-local-share-icons", label: ".local/share/icons", path: ($home_dir + "/.local/share/icons"), type: "sendreceive", rescanIntervalS: 120, fsWatcherEnabled: true, fsWatcherDelayS: 5, ignorePerms: false, ignoreDelete: false, autoNormalize: true, paused: false,
      devices: [{deviceID:$desktop_id,introducedBy:"",encryptionPassword:""},{deviceID:$laptop_id,introducedBy:"",encryptionPassword:""}]}) as $f10
  | ensure_device($laptopDevice)
  | ensure_folder($f1)
  | ensure_folder($f2)
  | ensure_folder($f3)
  | ensure_folder($f4)
  | ensure_folder($f5)
  | ensure_folder($f6)
  | ensure_folder($f7)
  | ensure_folder($f8)
  | ensure_folder($f9)
  | ensure_folder($f10)
  ' "$CFG_JSON" > "$NEW_CFG_JSON"

curl -fsS -X PUT \
  -H "X-API-Key: $API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @"$NEW_CFG_JSON" \
  "$BASE_URL/rest/config" >/dev/null

curl -fsS -X POST -H "X-API-Key: $API_KEY" "$BASE_URL/rest/system/restart" >/dev/null || true
sleep 5

API_KEY="$(sed -n 's:.*<apikey>\(.*\)</apikey>.*:\1:p' "$CFG_XML")"
for fid in dot-eidosian dot-ruff-cache dot-benchmarks dot-gnupg dot-config dot-codex dot-local-share-backgrounds dot-local-share-wallpapers dot-local-share-plasma dot-local-share-icons; do
  curl -fsS -X POST -H "X-API-Key: $API_KEY" "$BASE_URL/rest/db/scan?folder=${fid}" >/dev/null || true
done

echo "Desktop bidirectional sync config applied."
