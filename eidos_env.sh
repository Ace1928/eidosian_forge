#!/bin/bash
# 💎 EIDOSIAN FORGE ENVIRONMENT MODULE ⚡
# "Precision is functional alignment; integration is perfection."

export FORGE_ROOT="/data/data/com.termux/files/home/eidosian_forge"
export VENV_PATH="$FORGE_ROOT/eidosian_venv"

# Always prioritize the project virtual environment when available.
if [ -x "$VENV_PATH/bin/python" ]; then
    [[ ":$PATH:" != *":$VENV_PATH/bin:"* ]] && export PATH="$VENV_PATH/bin:$PATH"
    export VIRTUAL_ENV="$VENV_PATH"
fi

# PATH updates for all forge bin directories
for forge_dir in "$FORGE_ROOT"/*_forge "$FORGE_ROOT"/lib "$FORGE_ROOT"/eidos_mcp; do
    if [ -d "$forge_dir/bin" ]; then
        [[ ":$PATH:" != *":$forge_dir/bin:"* ]] && export PATH="$forge_dir/bin:$PATH"
    fi
done

# General Eidosian bin
[[ ":$PATH:" != *":$FORGE_ROOT/bin:"* ]] && export PATH="$FORGE_ROOT/bin:$PATH"
[[ -d "$FORGE_ROOT/llama.cpp/build/bin" ]] && [[ ":$PATH:" != *":$FORGE_ROOT/llama.cpp/build/bin:"* ]] && export PATH="$FORGE_ROOT/llama.cpp/build/bin:$PATH"
[[ -d "$FORGE_ROOT/llm_forge/bin" ]] && [[ ":$PATH:" != *":$FORGE_ROOT/llm_forge/bin:"* ]] && export PATH="$FORGE_ROOT/llm_forge/bin:$PATH"

# PYTHONPATH updates for all forge src directories (Editable installs handle this, but explicit is better for discovery)
_eidos_pythonpath="${PYTHONPATH:-}"
for forge_dir in "$FORGE_ROOT"/*_forge "$FORGE_ROOT"/lib "$FORGE_ROOT"/eidos_mcp; do
    if [ -d "$forge_dir/src" ]; then
        [[ ":$_eidos_pythonpath:" != *":$forge_dir/src:"* ]] && _eidos_pythonpath="$forge_dir/src:${_eidos_pythonpath}"
    fi
    # Include the root of the forge too for non-src layouts
    [[ ":$_eidos_pythonpath:" != *":$forge_dir:"* ]] && _eidos_pythonpath="$forge_dir:${_eidos_pythonpath}"
done
export PYTHONPATH="${_eidos_pythonpath}"
unset _eidos_pythonpath

# Dynamic linker paths for local llama.cpp toolchain.
for llama_lib_dir in "$FORGE_ROOT/llama.cpp/build/bin" "$FORGE_ROOT/llm_forge/vendor/llama.cpp/build/bin"; do
    if [ -d "$llama_lib_dir" ]; then
        if [ -n "${LD_LIBRARY_PATH:-}" ]; then
            [[ ":$LD_LIBRARY_PATH:" != *":$llama_lib_dir:"* ]] && export LD_LIBRARY_PATH="$llama_lib_dir:$LD_LIBRARY_PATH"
        else
            export LD_LIBRARY_PATH="$llama_lib_dir"
        fi
    fi
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

# Notification (disabled for non-interactive/service contexts)
if [ -z "${EIDOS_DISABLE_NOTIFICATIONS:-}" ] && [ -n "${PS1:-}" ] && command -v termux-notification >/dev/null 2>&1; then
    termux-notification --title "Eidosian Nexus" --content "Environment Module Loaded" --id 100 --priority low
fi
