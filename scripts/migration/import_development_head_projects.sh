#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  import_development_head_projects.sh [options]

Imports selected high-value project families from Development git HEAD into
existing eidosian_forge targets without overwriting existing files.

Options:
  --development-repo <path>  Development repo path (default: /home/lloyd/Development)
  --forge-root <path>        Forge root path (default: /home/lloyd/eidosian_forge)
  --tag <value>              Tag for output/manifests (default: timestamp)
  --execute                  Perform import (default: preview only)
  --help                     Show this help text

Behavior:
  - Source is extracted from git HEAD via `git archive` (independent of working tree deletions).
  - Destination sync uses `rsync --ignore-existing`.
  - Existing files are never overwritten.
EOF
}

DEV_REPO="/home/lloyd/Development"
FORGE_ROOT="/home/lloyd/eidosian_forge"
TAG="$(date +%Y-%m-%d_%H%M%S)"
EXECUTE="no"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --development-repo) DEV_REPO="${2:-}"; shift 2 ;;
    --forge-root) FORGE_ROOT="${2:-}"; shift 2 ;;
    --tag) TAG="${2:-}"; shift 2 ;;
    --execute) EXECUTE="yes"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -d "$DEV_REPO/.git" ]] || { echo "Not a git repo: $DEV_REPO" >&2; exit 1; }
[[ -d "$FORGE_ROOT" ]] || { echo "Forge root not found: $FORGE_ROOT" >&2; exit 1; }

MANIFEST_DIR="$FORGE_ROOT/archive_forge/manifests/development_archeology_${TAG}"
mkdir -p "$MANIFEST_DIR"
PLAN_FILE="$MANIFEST_DIR/development_head_import_plan.txt"
RESULT_FILE="$MANIFEST_DIR/development_head_import_results.txt"

# source_path_in_development -> destination_path_in_forge
declare -A MAP=()
MAP[ECosmos]="game_forge/src/ECosmos"
MAP[chess_game]="game_forge/src/chess_game"
MAP[EMemory]="memory_forge/legacy_imports/development_head_${TAG}/EMemory"
MAP[python_repository]="llm_forge/legacy_imports/development_head_${TAG}/python_repository"
MAP[oumi-main]="llm_forge/legacy_imports/development_head_${TAG}/oumi-main"
MAP[eidos_framework]="llm_forge/legacy_imports/development_head_${TAG}/eidos_framework"
MAP[templates]="prompt_forge/legacy_imports/development_head_${TAG}/templates"
MAP[notebooks]="doc_forge/legacy_imports/development_head_${TAG}/notebooks"
MAP[papers]="doc_forge/legacy_imports/development_head_${TAG}/papers"

items=(
  ECosmos
  chess_game
  EMemory
  python_repository
  oumi-main
  eidos_framework
  templates
  notebooks
  papers
)

{
  echo "timestamp=$(date -Iseconds)"
  echo "mode=$([[ "$EXECUTE" == "yes" ]] && echo execute || echo preview)"
  echo "development_repo=$DEV_REPO"
  echo "forge_root=$FORGE_ROOT"
  echo "tag=$TAG"
  echo
  printf '%-18s | %-9s | %-9s | %s\n' "source" "files" "size" "destination"
  printf '%-18s-+-%-9s-+-%-9s-+-%s\n' "------------------" "---------" "---------" "------------------------------"
} > "$PLAN_FILE"

tmp_src="$(mktemp -d)"
cleanup() {
  set +e
  rm -rf "$tmp_src"
}
trap cleanup EXIT

for src in "${items[@]}"; do
  if ! git -C "$DEV_REPO" ls-tree -d --name-only HEAD -- "$src" | grep -q "^${src}$"; then
    continue
  fi

  mkdir -p "$tmp_src/$src"
  git -C "$DEV_REPO" archive --format=tar HEAD "$src" | tar -xf - -C "$tmp_src"

  files=$(find "$tmp_src/$src" -type f | wc -l)
  size=$(du -sh "$tmp_src/$src" | awk '{print $1}')
  printf '%-18s | %-9s | %-9s | %s\n' "$src" "$files" "$size" "${MAP[$src]}" >> "$PLAN_FILE"
done

if [[ "$EXECUTE" != "yes" ]]; then
  echo "Preview complete: $PLAN_FILE"
  exit 0
fi

{
  echo "timestamp=$(date -Iseconds)"
  echo "mode=execute"
  echo "development_repo=$DEV_REPO"
  echo "forge_root=$FORGE_ROOT"
  echo "tag=$TAG"
  echo
  printf '%-18s | %-11s | %-11s | %-11s | %s\n' "source" "src_files" "transferred" "skipped" "destination"
  printf '%-18s-+-%-11s-+-%-11s-+-%-11s-+-%s\n' "------------------" "-----------" "-----------" "-----------" "------------------------------"
} > "$RESULT_FILE"

for src in "${items[@]}"; do
  [[ -d "$tmp_src/$src" ]] || continue
  dest="$FORGE_ROOT/${MAP[$src]}"
  mkdir -p "$dest"

  src_files=$(find "$tmp_src/$src" -type f | wc -l)
  stats_tmp=$(mktemp)
  rsync -a --ignore-existing --stats "$tmp_src/$src/" "$dest/" > "$stats_tmp"

  transferred=$(awk -F: '/Number of regular files transferred/{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' "$stats_tmp" | tr -d ',')
  [[ -n "$transferred" ]] || transferred=0
  skipped=$((src_files - transferred))

  printf '%-18s | %-11s | %-11s | %-11s | %s\n' "$src" "$src_files" "$transferred" "$skipped" "${MAP[$src]}" >> "$RESULT_FILE"
  rm -f "$stats_tmp"
done

echo "Plan:    $PLAN_FILE"
echo "Results: $RESULT_FILE"
