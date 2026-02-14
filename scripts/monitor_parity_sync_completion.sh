#!/usr/bin/env bash
set -euo pipefail

REMOTE_ID="${1:-4VUP2VH-MU76KJV-NYU32TL-YTXAZMZ-BRP6V3G-ISJWKVL-6K5DOS7-7EQYEAV}"
CFG_XML="/home/lloyd/.local/state/syncthing/config.xml"

API_KEY="$(sed -n 's:.*<apikey>\(.*\)</apikey>.*:\1:p' "$CFG_XML")"
if [[ -z "$API_KEY" ]]; then
  echo "Failed to read Syncthing API key from $CFG_XML" >&2
  exit 1
fi

BASE="http://127.0.0.1:8384"

while true; do
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  connected="$(curl -fsS -H "X-API-Key: $API_KEY" "$BASE/rest/system/connections" \
    | jq -r --arg rid "$REMOTE_ID" '.connections[$rid].connected // false')"
  echo "[$now] connected=$connected remote=$REMOTE_ID"

  mapfile -t folders < <(
    curl -fsS -H "X-API-Key: $API_KEY" "$BASE/rest/config/folders" \
      | jq -r '.[] | select(.id|test("^(dot-.*|home-pictures)$")) | .id' \
      | sort
  )

  all_done=1
  for folder in "${folders[@]}"; do
    row="$(curl -fsS -H "X-API-Key: $API_KEY" \
      "$BASE/rest/db/completion?folder=${folder}&device=${REMOTE_ID}")"
    completion="$(jq -r '.completion' <<<"$row")"
    need_bytes="$(jq -r '.needBytes' <<<"$row")"
    need_items="$(jq -r '.needItems' <<<"$row")"
    remote_state="$(jq -r '.remoteState' <<<"$row")"

    printf '  %-36s completion=%9s needBytes=%14s needItems=%8s remoteState=%s\n' \
      "$folder" "$completion" "$need_bytes" "$need_items" "$remote_state"

    if [[ "$completion" != "100" && "$completion" != "100.0" ]]; then
      all_done=0
    fi
  done

  if [[ "$all_done" -eq 1 ]]; then
    echo "All parity folders reached 100% completion."
    exit 0
  fi

  echo
  sleep 10
done
