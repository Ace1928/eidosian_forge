#!/usr/bin/env bash
# Eidosian Forge shell bootstrap.
# Keeps startup modular by loading environment parts from shell/env.d.

export FORGE_ROOT="/data/data/com.termux/files/home/eidosian_forge"
export VENV_PATH="$FORGE_ROOT/eidosian_venv"
export EIDOS_SHELL_MODULE_DIR="$FORGE_ROOT/shell/env.d"

if [ ! -d "$EIDOS_SHELL_MODULE_DIR" ]; then
    printf 'Missing Eidosian shell module dir: %s\n' "$EIDOS_SHELL_MODULE_DIR" >&2
    return 1 2>/dev/null || exit 1
fi

for eidos_module in "$EIDOS_SHELL_MODULE_DIR"/*.sh; do
    [ -f "$eidos_module" ] || continue
    # shellcheck source=/dev/null
    source "$eidos_module"
done

unset eidos_module
