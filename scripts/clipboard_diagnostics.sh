#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="clipboard_diagnostics"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/bash_lib.sh
. "$SCRIPT_DIR/lib/bash_lib.sh"

usage() {
  cat <<USAGE
Usage: $SCRIPT_NAME [--json] [--show-sample]

Detect available clipboard tools and test read access.

Options:
  --json          Output machine-readable JSON
  --show-sample   Print a sample of clipboard contents
  -h, --help      Show this help
USAGE
}

OUTPUT_JSON=false
SHOW_SAMPLE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      OUTPUT_JSON=true
      shift
      ;;
    --show-sample)
      SHOW_SAMPLE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

declare -A results

test_backend() {
  local name="$1"
  local cmd="$2"
  shift 2
  local ok=false
  local sample=""

  if ! command -v "$cmd" >/dev/null 2>&1; then
    results["$name"]="missing"
    return
  fi

  if $SHOW_SAMPLE; then
    sample=$("$cmd" "$@" 2>/dev/null | head -c 50 || true)
  else
    "$cmd" "$@" >/dev/null 2>&1 && ok=true || ok=false
  fi

  if $SHOW_SAMPLE; then
    if [[ -n "$sample" ]]; then
      results["$name"]="ok:$sample"
    else
      results["$name"]="fail"
    fi
  else
    if $ok; then
      results["$name"]="ok"
    else
      results["$name"]="fail"
    fi
  fi
}

if $OUTPUT_JSON; then
  test_backend "qdbus" qdbus org.kde.klipper /klipper org.kde.klipper.klipper.getClipboardContents
  test_backend "wl-paste" wl-paste
  test_backend "xclip" xclip -selection clipboard -o
  test_backend "xsel" xsel -b -o

  printf '{'
  first=true
  for key in qdbus wl-paste xclip xsel; do
    if ! $first; then
      printf ', '
    fi
    first=false
    printf '"%s": "%s"' "$key" "${results[$key]}"
  done
  printf '}'
  printf '\n'
  exit 0
fi

log_info "Clipboard diagnostics"
log_info "Tip: copy some text before running."

test_backend "xsel" xsel -b -o
test_backend "xclip" xclip -selection clipboard -o
test_backend "qdbus" qdbus org.kde.klipper /klipper org.kde.klipper.klipper.getClipboardContents
test_backend "wl-paste" wl-paste

for key in "${!results[@]}"; do
  case "${results[$key]}" in
    ok)
      log_info "$key: ok"
      ;;
    fail)
      log_warn "$key: cannot read clipboard"
      ;;
    missing)
      log_warn "$key: not installed"
      ;;
    ok:*)
      log_info "$key: ok (sample: ${results[$key]#ok:}...)"
      ;;
    *)
      log_warn "$key: ${results[$key]}"
      ;;
  esac
 done
