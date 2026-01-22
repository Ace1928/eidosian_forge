#!/usr/bin/env bash
# sync_forges.sh
# Eidosian Forge - Environment Synchronization Script
#
# Purpose: Iterates through all sub-projects in the Forge and installs them
# into the canonical virtual environment in editable mode.
# Handles errors gracefully and provides a summary report.

set -u

# Configuration
FORGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$FORGE_ROOT/eidosian_venv"
LOG_FILE="$FORGE_ROOT/logs/sync_forges.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    local level=$1
    shift
    local msg="$*"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "${timestamp} [${level}] ${msg}" >> "$LOG_FILE"
    case $level in
        INFO) echo -e "${BLUE}[INFO]${NC} $msg" ;;
        SUCCESS) echo -e "${GREEN}[OK]${NC} $msg" ;;
        WARN) echo -e "${YELLOW}[WARN]${NC} $msg" ;;
        ERROR) echo -e "${RED}[ERR]${NC} $msg" ;;
    esac
}

# 1. Activate Canonical Environment
log "INFO" "Checking canonical environment at: $VENV_PATH"

if [ ! -d "$VENV_PATH" ]; then
    log "ERROR" "Canonical venv not found at $VENV_PATH"
    log "INFO" "Please run: python3 -m venv $VENV_PATH"
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
log "SUCCESS" "Activated virtual environment."

# 2. Upgrade Core Tools
log "INFO" "Upgrading pip and build tools..."
if pip install --upgrade pip wheel setuptools >/dev/null 2>&1; then
    log "SUCCESS" "Pip, wheel, and setuptools are up to date."
else
    log "WARN" "Failed to upgrade core tools. Proceeding anyway."
fi

# 3. Discovery
log "INFO" "Scanning for projects..."
projects=()
while IFS= read -r file; do
    projects+=("$(dirname "$file")")
done < <(find "$FORGE_ROOT" -maxdepth 2 -name "pyproject.toml" -not -path "*/.*" -not -path "$VENV_PATH*")

log "INFO" "Found ${#projects[@]} projects to install."

# 4. Installation Loop
failed_projects=()
success_count=0

for project_dir in "${projects[@]}"; do
    project_name=$(basename "$project_dir")
    log "INFO" "Installing $project_name..."

    # Capture output for debugging on failure
    if install_output=$(pip install -e "$project_dir" 2>&1); then
        log "SUCCESS" "Installed $project_name"
        ((success_count++))
    else
        log "ERROR" "Failed to install $project_name"
        echo "$install_output" >> "$LOG_FILE"
        failed_projects+=("$project_name")
    fi
done

# 5. Summary
echo "---------------------------------------------------"
log "INFO" "Synchronization Complete"
echo "---------------------------------------------------"
log "INFO" "Total Projects: ${#projects[@]}"
log "INFO" "Successful:     $success_count"
log "INFO" "Failed:         ${#failed_projects[@]}"

if [ ${#failed_projects[@]} -gt 0 ]; then
    echo -e "${RED}The following projects failed to install:${NC}"
    for fp in "${failed_projects[@]}"; do
        echo -e " - $fp"
    done
    echo "Check $LOG_FILE for details."
    exit 1
else
    log "SUCCESS" "All forges synced successfully."
    exit 0
fi
