#!/bin/bash
# Eidosian Documentation Forge Orchestrator
# Runs the documentation pipeline continuously until completion.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$FORGE_ROOT"

LOG_FILE="doc_forge/forge_run.log"

echo "Starting Eidosian Documentation Forge..." | tee -a "$LOG_FILE"

# 1. Generate Index if missing
if [ ! -f "doc_forge/file_index.json" ]; then
    echo "Generating file index..." | tee -a "$LOG_FILE"
    ./doc_forge/scripts/generate_docs.py >> "$LOG_FILE" 2>&1
fi

# 2. Main Processing Loop
echo "Starting processing loop..." | tee -a "$LOG_FILE"
# Run process_files.py with no limit (0). It handles resumption via file checks.
./doc_forge/scripts/process_files.py --limit 0 >> "$LOG_FILE" 2>&1

PROCESS_EXIT_CODE=$?

if [ $PROCESS_EXIT_CODE -eq 0 ]; then
    echo "Processing complete." | tee -a "$LOG_FILE"
else
    echo "Processing stopped with exit code $PROCESS_EXIT_CODE. Check logs." | tee -a "$LOG_FILE"
fi

# 3. Auto-Approve & Index
echo "Running auto-approval and indexing..." | tee -a "$LOG_FILE"
./doc_forge/scripts/auto_approve.py >> "$LOG_FILE" 2>&1
./doc_forge/scripts/generate_html_index.py >> "$LOG_FILE" 2>&1

echo "Forge run finished." | tee -a "$LOG_FILE"
