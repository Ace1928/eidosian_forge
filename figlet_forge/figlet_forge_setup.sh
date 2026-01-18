#!/usr/bin/env bash
#
# ╔═══════════════════════════════════════╗
# ║         FIGLET FORGE SETUP            ║
# ║         Eidosian Framework            ║
# ╚═══════════════════════════════════════╝
#
# Creates the Figlet Forge project structure with idempotency.
# Version: 1.0.0

set -o errexit
set -o pipefail
set -o nounset

# ═════════════════════ ESSENCE FUNCTIONS ═════════════════════

# Manifest directory with validation
manifest_dir() {
  local path="$1"
  [[ -d "$path" ]] && echo "⟲  Directory exists: $path" && return 0
  if mkdir -p "$path"; then
    echo "✓  Manifested: $path"
  else
    echo "✗  Failed to manifest: $path"
    return 1
  fi
}

manifest_file() {
  local path="$1"
  [[ -f "$path" ]] && echo "⟲  File exists: $path" && return 0
  if touch "$path"; then
    echo "✓  Manifested: $path"
  else
    echo "✗  Failed to manifest: $path"
    return 1
  fi
}

# ═════════════════════ FORGE STRUCTURE ═════════════════════
# Root paths
ROOT="figlet_forge"
DIRS=(
  "docs/{api,examples,fonts}"
  "src/figlet_forge/{core,color,render,fonts/{standard,contrib},compat,cli}"
  "tests/{unit,integration,compat,fonts}"
  "tools"
  "examples"
)

# Check if package structure exists and create it if it doesn't
# This ensures the package can be imported during development
if [ ! -d "$ROOT/src/figlet_forge" ]; then
  echo "Creating package structure for development..."
  mkdir -p "$ROOT/src/figlet_forge"
  # Create __init__.py files to make the packages importable
  touch "$ROOT/src/__init__.py"
  touch "$ROOT/src/figlet_forge/__init__.py"
fi

# Manifest directories
for dir_pattern in "${DIRS[@]}"; do
  for dir in $(eval echo "$ROOT/$dir_pattern"); do
    manifest_dir "$dir" || exit 1
  done
done

# Manifest Python module files
MODULES=(
  "src/figlet_forge/__init__.py"
  "src/figlet_forge/{core,color,render,fonts,compat,cli}/__init__.py"
  "tests/__init__.py"
  "tests/{unit,integration,compat,fonts}/__init__.py"
)

for file_pattern in "${MODULES[@]}"; do
  for file in $(eval echo "$ROOT/$file_pattern"); do
    manifest_file "$file" || exit 1
  done
done

echo "✨ Figlet Forge structure has been forged successfully."
