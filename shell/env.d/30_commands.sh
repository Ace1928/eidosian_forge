#!/usr/bin/env bash

python() {
    if _eidos_should_activate && [ -x "$VENV_PATH/bin/python" ]; then
        eidos_safe_exec "$VENV_PATH/bin/python" "$@"
    else
        command python "$@"
    fi
}

pip() {
    if _eidos_should_activate && [ -x "$VENV_PATH/bin/pip" ]; then
        eidos_safe_exec "$VENV_PATH/bin/pip" "$@"
    else
        command pip "$@"
    fi
}

alias forge='cd "$FORGE_ROOT"'
alias forge-env='cd "$FORGE_ROOT" && eidos_use_env'
alias forge-reset='eidos_reset_env'
alias nexus='forge'
alias eidos='forge'
alias st='git status'
alias precision='echo "Precision is functional alignment."'
alias strike='echo "Flow and Strike: precision execution."'
