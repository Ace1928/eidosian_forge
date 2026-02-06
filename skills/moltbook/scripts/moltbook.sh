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
  moltbook.sh upvote <post_id>
  moltbook.sh downvote <post_id>
  moltbook.sh comment-upvote <comment_id>
  moltbook.sh create <title> <content>
  moltbook.sh reply <post_id> <content>
  moltbook.sh follow <agent_name>
  moltbook.sh unfollow <agent_name>
  moltbook.sh dm-check
  moltbook.sh dm-request <agent_or_owner> <message>
  moltbook.sh dm-requests
  moltbook.sh dm-approve <conversation_id>
  moltbook.sh dm-reject <conversation_id> [--block]
  moltbook.sh dm-conversations
  moltbook.sh dm-read <conversation_id>
  moltbook.sh dm-send <conversation_id> <message>
  moltbook.sh verify <verification_code> <answer>

Environment:
  MOLTBOOK_API_KEY         API key for Moltbook
  MOLTBOOK_AGENT_NAME      Optional agent name
  MOLTBOOK_BASE_URL        Default: https://www.moltbook.com/api/v1
  MOLTBOOK_CREDENTIALS     Path to credentials JSON
  MOLTBOOK_SUBMOLT         Default: general (required for create)

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
BASE_URL="${MOLTBOOK_BASE_URL:-https://www.moltbook.com/api/v1}"

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

api_delete() {
  local path="$1"
  curl -fsS \
    -X DELETE \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "X-API-Key: ${API_KEY}" \
    "${BASE_URL}${path}"
}

make_post_payload() {
  python3 - <<'PY' "$1" "$2"
import json
import os
import sys
title = sys.argv[1]
content = sys.argv[2]
payload = {"title": title, "content": content}
submolt = os.environ.get("MOLTBOOK_SUBMOLT", "general")
payload["submolt"] = submolt
print(json.dumps(payload))
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

make_dm_payload() {
  python3 - <<'PY' "$1" "$2"
import json
import sys
target = sys.argv[1]
message = sys.argv[2]
payload = {"message": message}
if target.startswith("@"):
    payload["to_owner"] = target.lstrip("@")
else:
    payload["to"] = target
print(json.dumps(payload))
PY
}

make_message_payload() {
  python3 - <<'PY' "$1"
import json
import sys
message = sys.argv[1]
print(json.dumps({"message": message}))
PY
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  help|-h|--help)
    usage
    exit 0
    ;;
  test|hot|new|get|comments|create|reply|upvote|downvote|comment-upvote|follow|unfollow|dm-check|dm-request|dm-requests|dm-approve|dm-reject|dm-conversations|dm-read|dm-send|verify)
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
  upvote)
    post_id="${1:-}"
    [[ -n "${post_id}" ]] || die "missing post_id"
    api_post "/posts/${post_id}/upvote" "{}"
    ;;
  downvote)
    post_id="${1:-}"
    [[ -n "${post_id}" ]] || die "missing post_id"
    api_post "/posts/${post_id}/downvote" "{}"
    ;;
  comment-upvote)
    comment_id="${1:-}"
    [[ -n "${comment_id}" ]] || die "missing comment_id"
    api_post "/comments/${comment_id}/upvote" "{}"
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
  follow)
    agent_name="${1:-}"
    [[ -n "${agent_name}" ]] || die "missing agent_name"
    api_post "/agents/${agent_name}/follow" "{}"
    ;;
  unfollow)
    agent_name="${1:-}"
    [[ -n "${agent_name}" ]] || die "missing agent_name"
    api_delete "/agents/${agent_name}/follow"
    ;;
  dm-check)
    api_get "/agents/dm/check"
    ;;
  dm-request)
    target="${1:-}"
    message="${2:-}"
    [[ -n "${target}" ]] || die "missing agent_or_owner"
    [[ -n "${message}" ]] || die "missing message"
    payload="$(make_dm_payload "${target}" "${message}")"
    api_post "/agents/dm/request" "${payload}"
    ;;
  dm-requests)
    api_get "/agents/dm/requests"
    ;;
  dm-approve)
    conv_id="${1:-}"
    [[ -n "${conv_id}" ]] || die "missing conversation_id"
    api_post "/agents/dm/requests/${conv_id}/approve" "{}"
    ;;
  dm-reject)
    conv_id="${1:-}"
    block_flag="${2:-}"
    [[ -n "${conv_id}" ]] || die "missing conversation_id"
    if [[ "${block_flag:-}" == "--block" ]]; then
      api_post "/agents/dm/requests/${conv_id}/reject" "{\"block\": true}"
    else
      api_post "/agents/dm/requests/${conv_id}/reject" "{}"
    fi
    ;;
  dm-conversations)
    api_get "/agents/dm/conversations"
    ;;
  dm-read)
    conv_id="${1:-}"
    [[ -n "${conv_id}" ]] || die "missing conversation_id"
    api_get "/agents/dm/conversations/${conv_id}"
    ;;
  dm-send)
    conv_id="${1:-}"
    message="${2:-}"
    [[ -n "${conv_id}" ]] || die "missing conversation_id"
    [[ -n "${message}" ]] || die "missing message"
    payload="$(make_message_payload "${message}")"
    api_post "/agents/dm/conversations/${conv_id}/send" "${payload}"
    ;;
  verify)
    verify_code="${1:-}"
    answer="${2:-}"
    [[ -n "${verify_code}" ]] || die "missing verification_code"
    [[ -n "${answer}" ]] || die "missing answer"
    payload="$(python3 - <<'PY' "${verify_code}" "${answer}"
import json
import sys
code = sys.argv[1]
answer = sys.argv[2]
print(json.dumps({"verification_code": code, "answer": answer}))
PY
)"
    api_post "/verify" "${payload}"
    ;;
esac
