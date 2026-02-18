#!/bin/bash
# 💎 EIDOSIAN FORGE ENVIRONMENT MODULE ⚡
# "Precision is functional alignment; integration is perfection."

export FORGE_ROOT="/data/data/com.termux/files/home/eidosian_forge"
export VENV_PATH="$FORGE_ROOT/eidosian_venv"

# PATH updates for all forge bin directories
for forge_dir in "$FORGE_ROOT"/*_forge "$FORGE_ROOT"/lib "$FORGE_ROOT"/eidos_mcp; do
    if [ -d "$forge_dir/bin" ]; then
        [[ ":$PATH:" != *":$forge_dir/bin:"* ]] && export PATH="$forge_dir/bin:$PATH"
    fi
done

# General Eidosian bin
[[ ":$PATH:" != *":$FORGE_ROOT/bin:"* ]] && export PATH="$FORGE_ROOT/bin:$PATH"

# PYTHONPATH updates for all forge src directories (Editable installs handle this, but explicit is better for discovery)
for forge_dir in "$FORGE_ROOT"/*_forge "$FORGE_ROOT"/lib "$FORGE_ROOT"/eidos_mcp; do
    if [ -d "$forge_dir/src" ]; then
        [[ ":$PYTHONPATH:" != *":$forge_dir/src:"* ]] && export PYTHONPATH="$forge_dir/src:$PYTHONPATH"
    fi
    # Include the root of the forge too for non-src layouts
    [[ ":$PYTHONPATH:" != *":$forge_dir:"* ]] && export PYTHONPATH="$forge_dir:$PYTHONPATH"
done

# Eidosian Venv Wrapper
python() {
    if [[ "$PWD" == "$FORGE_ROOT"* ]] && [ -d "$VENV_PATH" ]; then
        "$VENV_PATH/bin/python" "$@"
    else
        command python "$@"
    fi
}

pip() {
    if [[ "$PWD" == "$FORGE_ROOT"* ]] && [ -d "$VENV_PATH" ]; then
        "$VENV_PATH/bin/pip" "$@"
    else
        command pip "$@"
    fi
}

# Advanced Eidosian Aliases
alias forge="cd $FORGE_ROOT"
alias nexus='forge'
alias eidos='forge'
alias st='git status'
alias precision='echo "💎 Precision is functional alignment."'
alias strike='echo "⚡ Flow and Strike: Precision execution."'

# Notification
if command -v termux-notification >/dev/null 2>&1; then
    termux-notification --title "Eidosian Nexus" --content "Environment Module Loaded" --id 100 --priority low
fi
