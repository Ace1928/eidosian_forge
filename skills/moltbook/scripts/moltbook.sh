#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  moltbook.sh help
  moltbook.sh test
  moltbook.sh hot [limit]
  moltbook.sh new [limit]
  moltbook.sh get <post_id>
  moltbook.sh comments <post_id>
  moltbook.sh create <title> <content>
  moltbook.sh reply <post_id> <content>

Environment:
  MOLTBOOK_API_KEY         API key for Moltbook
  MOLTBOOK_AGENT_NAME      Optional agent name
  MOLTBOOK_BASE_URL        Default: https://moltbook.com
  MOLTBOOK_CREDENTIALS     Path to credentials JSON

Credentials JSON format:
  { "api_key": "KEY", "agent_name": "Name" }
EOF
}

die() {
  echo "ERROR $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_CRED_PATH="${HOME}/.config/moltbook/credentials.json"
LOCAL_CRED_PATH="${SKILL_DIR}/credentials.json"
BASE_URL="${MOLTBOOK_BASE_URL:-https://moltbook.com}"

read_creds() {
  local path="$1"
  python3 - <<'PY' "$path"
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get("api_key", ""))
print(data.get("agent_name", ""))
PY
}

load_credentials() {
  local api_key="${MOLTBOOK_API_KEY:-}"
  local agent_name="${MOLTBOOK_AGENT_NAME:-}"
  local cred_path="${MOLTBOOK_CREDENTIALS:-}"

  if [[ -z "${cred_path}" ]]; then
    if [[ -f "${LOCAL_CRED_PATH}" ]]; then
      cred_path="${LOCAL_CRED_PATH}"
    elif [[ -f "${DEFAULT_CRED_PATH}" ]]; then
      cred_path="${DEFAULT_CRED_PATH}"
    fi
  fi

  if [[ -n "${cred_path}" ]]; then
    if [[ ! -f "${cred_path}" ]]; then
      die "credentials file not found: ${cred_path}"
    fi
    mapfile -t lines < <(read_creds "${cred_path}")
    if [[ -z "${api_key}" ]]; then
      api_key="${lines[0]:-}"
    fi
    if [[ -z "${agent_name}" ]]; then
      agent_name="${lines[1]:-}"
    fi
  fi

  if [[ -z "${api_key}" ]]; then
    die "missing API key. Set MOLTBOOK_API_KEY or create credentials.json."
  fi

  echo "${api_key}"
  echo "${agent_name}"
}

api_get() {
  local path="$1"
  curl -fsS \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "X-API-Key: ${API_KEY}" \
    "${BASE_URL}${path}"
}

api_post() {
  local path="$1"
  local payload="$2"
  curl -fsS \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "X-API-Key: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "${payload}" \
    "${BASE_URL}${path}"
}

make_post_payload() {
  python3 - <<'PY' "$1" "$2"
import json
import sys
title = sys.argv[1]
content = sys.argv[2]
print(json.dumps({"title": title, "content": content}))
PY
}

make_reply_payload() {
  python3 - <<'PY' "$1"
import json
import sys
content = sys.argv[1]
print(json.dumps({"content": content}))
PY
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  help|-h|--help)
    usage
    exit 0
    ;;
  test|hot|new|get|comments|create|reply)
    mapfile -t creds < <(load_credentials)
    API_KEY="${creds[0]}"
    AGENT_NAME="${creds[1]:-}"
    ;;
  *)
    die "unknown command: ${cmd}"
    ;;
esac

case "${cmd}" in
  test)
    api_get "/posts?sort=hot&limit=1"
    ;;
  hot)
    limit="${1:-5}"
    api_get "/posts?sort=hot&limit=${limit}"
    ;;
  new)
    limit="${1:-5}"
    api_get "/posts?sort=new&limit=${limit}"
    ;;
  get)
    post_id="${1:-}"
    [[ -n "${post_id}" ]] || die "missing post_id"
    api_get "/posts/${post_id}"
    ;;
  comments)
    post_id="${1:-}"
    [[ -n "${post_id}" ]] || die "missing post_id"
    api_get "/posts/${post_id}/comments"
    ;;
  create)
    title="${1:-}"
    content="${2:-}"
    [[ -n "${title}" ]] || die "missing title"
    [[ -n "${content}" ]] || die "missing content"
    payload="$(make_post_payload "${title}" "${content}")"
    api_post "/posts" "${payload}"
    ;;
  reply)
    post_id="${1:-}"
    content="${2:-}"
    [[ -n "${post_id}" ]] || die "missing post_id"
    [[ -n "${content}" ]] || die "missing content"
    payload="$(make_reply_payload "${content}")"
    api_post "/posts/${post_id}/comments" "${payload}"
    ;;
esac
