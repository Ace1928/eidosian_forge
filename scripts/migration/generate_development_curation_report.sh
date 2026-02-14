#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  generate_development_curation_report.sh [options]

Generates an archaeology/curation report from Development git HEAD with:
  - project family inventory
  - destination mapping
  - canonical drift checks for ECosmos/chess_game

Options:
  --development-repo <path>  (default: /home/lloyd/Development)
  --forge-root <path>        (default: /home/lloyd/eidosian_forge)
  --tag <value>              (default: timestamp)
EOF
}

DEV_REPO="/home/lloyd/Development"
FORGE_ROOT="/home/lloyd/eidosian_forge"
TAG="$(date +%Y-%m-%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --development-repo) DEV_REPO="${2:-}"; shift 2 ;;
    --forge-root) FORGE_ROOT="${2:-}"; shift 2 ;;
    --tag) TAG="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -d "$DEV_REPO/.git" ]] || { echo "Not a git repo: $DEV_REPO" >&2; exit 1; }
[[ -d "$FORGE_ROOT" ]] || { echo "Forge root not found: $FORGE_ROOT" >&2; exit 1; }

OUT_DIR="$FORGE_ROOT/archive_forge/manifests/development_archeology_${TAG}"
mkdir -p "$OUT_DIR"
OUT_MD="$OUT_DIR/curation_report.md"

tmp="$(mktemp -d)"
cleanup() {
  set +e
  rm -rf "$tmp"
}
trap cleanup EXIT

git -C "$DEV_REPO" archive --format=tar HEAD ECosmos chess_game EMemory python_repository oumi-main eidos_framework templates notebooks papers \
  | tar -xf - -C "$tmp"

{
  echo "# Development Archaeology Curation Report"
  echo
  echo "- Timestamp: $(date -Iseconds)"
  echo "- Development HEAD: $(git -C "$DEV_REPO" rev-parse --short HEAD)"
  echo "- Forge root: $FORGE_ROOT"
  echo
  echo "## Coverage Matrix"
  echo
  echo "| Source | Files | Suggested Destination | Status |"
  echo "|---|---:|---|---|"
  for src in ECosmos chess_game EMemory python_repository oumi-main eidos_framework templates notebooks papers; do
    files=0
    [[ -d "$tmp/$src" ]] && files=$(find "$tmp/$src" -type f | wc -l)
    dest="unmapped"
    status="present"
    case "$src" in
      ECosmos) dest="game_forge/src/ECosmos";;
      chess_game) dest="game_forge/src/chess_game";;
      EMemory) dest="memory_forge/legacy_imports/development_head_${TAG}/EMemory";;
      python_repository) dest="llm_forge/legacy_imports/development_head_${TAG}/python_repository";;
      oumi-main) dest="llm_forge/legacy_imports/development_head_${TAG}/oumi-main";;
      eidos_framework) dest="llm_forge/legacy_imports/development_head_${TAG}/eidos_framework";;
      templates) dest="prompt_forge/legacy_imports/development_head_${TAG}/templates";;
      notebooks) dest="doc_forge/legacy_imports/development_head_${TAG}/notebooks";;
      papers) dest="doc_forge/legacy_imports/development_head_${TAG}/papers";;
    esac
    [[ -d "$tmp/$src" ]] || status="missing_in_head"
    echo "| $src | $files | \`$dest\` | $status |"
  done
  echo
  echo "## Canonical Drift Check"
  echo
  echo "### ECosmos"
  diff -rq "$tmp/ECosmos" "$FORGE_ROOT/game_forge/src/ECosmos" \
    | sed "s|$tmp/ECosmos|Development_HEAD/ECosmos|g" \
    | sed "s|$FORGE_ROOT/game_forge/src/ECosmos|Forge/game_forge/src/ECosmos|g" \
    | sed 's/^/- /' || true
  echo
  echo "### chess_game"
  diff -rq "$tmp/chess_game" "$FORGE_ROOT/game_forge/src/chess_game" \
    | sed "s|$tmp/chess_game|Development_HEAD/chess_game|g" \
    | sed "s|$FORGE_ROOT/game_forge/src/chess_game|Forge/game_forge/src/chess_game|g" \
    | sed 's/^/- /' || true
  echo
  echo "## Recommended Next Upgrades"
  echo
  echo "1. Create a deterministic merge pass for ECosmos/chess_game (3-way merge against Development HEAD)."
  echo "2. Add smoke tests for imported EMemory and python_repository entry points."
  echo "3. Promote stable notebooks/papers into doc_forge index manifests with topic tags."
  echo "4. Keep raw legacy imports immutable under timestamped directories; build curated derivatives in forge-native paths."
} > "$OUT_MD"

echo "$OUT_MD"
