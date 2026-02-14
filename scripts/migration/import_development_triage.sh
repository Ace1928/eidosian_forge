#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: import_development_triage.sh [options]

Imports /home/lloyd/Development/_triage buckets into existing eidosian_forge
repositories using non-destructive rsync + manifest logging.

Options:
  --source <path>        Source triage root (default: /home/lloyd/Development/_triage)
  --forge-root <path>    Forge root (default: /home/lloyd/eidosian_forge)
  --tag <value>          Migration tag (default: current timestamp)
  --execute              Perform copy operations (default is preview only)
  --help                 Show this help text

Behavior:
  - Preview mode writes a plan + estimated file counts only.
  - Execute mode uses rsync --ignore-existing and writes per-bucket stats.
  - Existing files are never overwritten.
USAGE
}

SOURCE_ROOT="/home/lloyd/Development/_triage"
FORGE_ROOT="/home/lloyd/eidosian_forge"
TAG="$(date +%Y-%m-%d_%H%M%S)"
EXECUTE="no"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_ROOT="${2:-}"
      shift 2
      ;;
    --forge-root)
      FORGE_ROOT="${2:-}"
      shift 2
      ;;
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --execute)
      EXECUTE="yes"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Source root not found: $SOURCE_ROOT" >&2
  exit 1
fi
if [[ ! -d "$FORGE_ROOT" ]]; then
  echo "Forge root not found: $FORGE_ROOT" >&2
  exit 1
fi

MANIFEST_DIR="$FORGE_ROOT/archive_forge/manifests/development_archeology_${TAG}"
mkdir -p "$MANIFEST_DIR"
PLAN_FILE="$MANIFEST_DIR/triage_import_plan.txt"
RESULT_FILE="$MANIFEST_DIR/triage_import_results.txt"

# bucket -> destination path (relative to forge root)
declare -A MAP=()
MAP[archive_forge]="archive_forge/legacy_imports/development_triage_${TAG}/archive_forge"
MAP[archive_misc]="archive_forge/legacy_imports/development_triage_${TAG}/archive_misc"
MAP[crawl_forge]="crawl_forge/legacy_imports/development_triage_${TAG}/crawl_forge"
MAP[data_forge_candidate]="projects/legacy/development_triage_${TAG}/data_forge_candidate"
MAP[doc_forge]="doc_forge/legacy_imports/development_triage_${TAG}/doc_forge"
MAP[game_forge]="game_forge/legacy_imports/development_triage_${TAG}/game_forge"
MAP[knowledge_forge]="knowledge_forge/legacy_imports/development_triage_${TAG}/knowledge_forge"
MAP[llm_forge]="llm_forge/legacy_imports/development_triage_${TAG}/llm_forge"
MAP[memory_forge]="memory_forge/legacy_imports/development_triage_${TAG}/memory_forge"
MAP[narrative_forge]="narrative_forge/legacy_imports/development_triage_${TAG}/narrative_forge"
MAP[prompt_forge]="prompt_forge/legacy_imports/development_triage_${TAG}/prompt_forge"
MAP[viz_forge]="viz_forge/legacy_imports/development_triage_${TAG}/viz_forge"

{
  echo "timestamp=$(date -Iseconds)"
  echo "mode=$([[ "$EXECUTE" == "yes" ]] && echo execute || echo preview)"
  echo "source_root=$SOURCE_ROOT"
  echo "forge_root=$FORGE_ROOT"
  echo "tag=$TAG"
  echo
  printf '%-22s | %-8s | %-10s | %s\n' "bucket" "files" "size" "destination"
  printf '%-22s-+-%-8s-+-%-10s-+-%s\n' "----------------------" "--------" "----------" "-----------------------------"
} > "$PLAN_FILE"

# Stable processing order.
buckets=(
  archive_forge
  archive_misc
  crawl_forge
  data_forge_candidate
  doc_forge
  game_forge
  knowledge_forge
  llm_forge
  memory_forge
  narrative_forge
  prompt_forge
  viz_forge
)

for bucket in "${buckets[@]}"; do
  src="$SOURCE_ROOT/$bucket"
  if [[ ! -d "$src" ]]; then
    continue
  fi
  dest_rel="${MAP[$bucket]}"
  dest="$FORGE_ROOT/$dest_rel"
  files=$(find "$src" -type f | wc -l)
  size=$(du -sh "$src" | awk '{print $1}')
  printf '%-22s | %-8s | %-10s | %s\n' "$bucket" "$files" "$size" "$dest_rel" >> "$PLAN_FILE"
done

if [[ "$EXECUTE" != "yes" ]]; then
  echo "Preview complete: $PLAN_FILE"
  exit 0
fi

{
  echo "timestamp=$(date -Iseconds)"
  echo "mode=execute"
  echo "source_root=$SOURCE_ROOT"
  echo "forge_root=$FORGE_ROOT"
  echo "tag=$TAG"
  echo
  printf '%-22s | %-11s | %-11s | %-11s | %s\n' "bucket" "src_files" "transferred" "skipped" "destination"
  printf '%-22s-+-%-11s-+-%-11s-+-%-11s-+-%s\n' "----------------------" "-----------" "-----------" "-----------" "-----------------------------"
} > "$RESULT_FILE"

for bucket in "${buckets[@]}"; do
  src="$SOURCE_ROOT/$bucket"
  [[ -d "$src" ]] || continue

  dest_rel="${MAP[$bucket]}"
  dest="$FORGE_ROOT/$dest_rel"
  mkdir -p "$dest"

  src_files=$(find "$src" -type f | wc -l)

  stats_tmp=$(mktemp)
  rsync -a --ignore-existing --stats "$src/" "$dest/" > "$stats_tmp"

  transferred=$(awk -F: '/Number of regular files transferred/{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' "$stats_tmp" | tr -d ',')
  if [[ -z "$transferred" ]]; then
    transferred=0
  fi
  skipped=$((src_files - transferred))
  printf '%-22s | %-11s | %-11s | %-11s | %s\n' "$bucket" "$src_files" "$transferred" "$skipped" "$dest_rel" >> "$RESULT_FILE"

  rm -f "$stats_tmp"
done

echo "Plan:    $PLAN_FILE"
echo "Results: $RESULT_FILE"
