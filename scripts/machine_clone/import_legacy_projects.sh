#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  import_legacy_projects.sh --source-root <path> --manifest-dir <path>

Description:
  Imports curated source/docs from legacy Aurora SSD layouts into eidosian_forge.
  This script is additive and skips existing target files.
EOF
}

SOURCE_ROOT=""
MANIFEST_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      SOURCE_ROOT="${2:-}"; shift 2 ;;
    --manifest-dir)
      MANIFEST_DIR="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "$SOURCE_ROOT" || -z "$MANIFEST_DIR" ]]; then
  echo "--source-root and --manifest-dir are required" >&2
  exit 1
fi

BASE="/home/lloyd/eidosian_forge"
STAMP="$(date +%Y-%m-%d_%H%M%S)"
RUN_DIR="${MANIFEST_DIR%/}/legacy_import_${STAMP}"
mkdir -p "$RUN_DIR"

SRC_EVIE="${SOURCE_ROOT%/}/lloyd/Documents/BackupStuffAgain/EVIE"
SRC_EXTRA="${SOURCE_ROOT%/}/lloyd/Documents/BackupStuffAgain/extra-repos"
SRC_DOCS="${SOURCE_ROOT%/}/lloyd/Documents/BackupStuffAgain"

DST_EVIE="$BASE/projects/legacy/evie"
DST_IND_SNAKE="$BASE/projects/legacy/indego_snake_game"
DST_MC="$BASE/projects/legacy/minecraft_ai_indego"
DST_AIT="$BASE/projects/legacy/ai_timeline"
DST_3D="$BASE/projects/legacy/3dstuff"
DST_DOCS="$BASE/docs/legacy/aurora_archive"

mkdir -p "$DST_EVIE" "$DST_IND_SNAKE" "$DST_MC" "$DST_AIT" "$DST_3D" "$DST_DOCS"

common_excludes=(
  --exclude '.git/'
  --exclude '.venv/'
  --exclude 'venv/'
  --exclude '__pycache__/'
  --exclude 'node_modules/'
  --exclude '*.log'
  --exclude '*.db'
  --exclude '*.sqlite*'
  --exclude '*.pt'
  --exclude '*.bin'
  --exclude '*.safetensors'
  --exclude '*.gguf'
  --exclude '*.tar'
  --exclude '*.tar.gz'
  --exclude '*.zip'
  --exclude '*.7z'
  --exclude '*.deb'
)

rsync -a --ignore-existing --prune-empty-dirs \
  "${common_excludes[@]}" \
  --exclude 'MetaAI/' \
  --exclude 'RWKV-Runner/' \
  --exclude 'indego/' \
  --exclude 'models/' \
  --exclude 'lora-models/' \
  --exclude 'build/' \
  --exclude 'py310/' \
  --exclude 'redundant/' \
  "$SRC_EVIE/" "$DST_EVIE/"

rsync -a --ignore-existing --prune-empty-dirs "${common_excludes[@]}" "$SRC_EXTRA/minecraft_ai_indego/" "$DST_MC/"
rsync -a --ignore-existing --prune-empty-dirs "${common_excludes[@]}" "$SRC_EXTRA/ai_timeline/" "$DST_AIT/"
rsync -a --ignore-existing --prune-empty-dirs "${common_excludes[@]}" "$SRC_EXTRA/3DStuff/" "$DST_3D/"

if [[ -d "$SRC_EXTRA/Indego-SnakeGame/.git" ]]; then
  git -C "$SRC_EXTRA/Indego-SnakeGame" archive --format=tar HEAD | tar -xf - -C "$DST_IND_SNAKE"
fi

DOC_HASH_TSV="$RUN_DIR/docs_hashes.tsv"
DOC_UNIQ_LIST="$RUN_DIR/docs_unique_paths.txt"
while IFS= read -r -d '' f; do
  h="$(sha256sum "$f" | awk '{print $1}')"
  printf '%s\t%s\n' "$h" "$f" >> "$DOC_HASH_TSV"
done < <(find "$SRC_DOCS" -maxdepth 1 -type f \( -iname '*.docx' -o -iname '*.odt' -o -iname '*.pdf' \) -print0)
awk -F'\t' '!seen[$1]++ {print $2}' "$DOC_HASH_TSV" > "$DOC_UNIQ_LIST"
while IFS= read -r f; do
  [[ -n "$f" ]] || continue
  cp --update=none "$f" "$DST_DOCS/"
done < "$DOC_UNIQ_LIST"

IMPORTED_NULL="$RUN_DIR/imported_files.null"
IMPORTED_SHA="$RUN_DIR/imported_sha256_manifest.txt"
find "$DST_EVIE" "$DST_IND_SNAKE" "$DST_MC" "$DST_AIT" "$DST_3D" "$DST_DOCS" -type f -print0 | sort -z > "$IMPORTED_NULL"
xargs -0 sha256sum < "$IMPORTED_NULL" > "$IMPORTED_SHA"

cat > "$RUN_DIR/legacy_import_manifest.v1.json" <<EOF
{
  "schema_version": "v1",
  "created_at": "${STAMP}",
  "source_root": "${SOURCE_ROOT}",
  "targets": {
    "evie": "${DST_EVIE}",
    "indego_snake_game": "${DST_IND_SNAKE}",
    "minecraft_ai_indego": "${DST_MC}",
    "ai_timeline": "${DST_AIT}",
    "3dstuff": "${DST_3D}",
    "docs": "${DST_DOCS}"
  }
}
EOF

echo "$RUN_DIR"
