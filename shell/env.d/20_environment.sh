#!/usr/bin/env bash

eidos_reset_env() {
    _eidos_capture_shell_baseline
    export PATH="${EIDOS_BASE_PATH:-${PATH:-}}"
    export PYTHONPATH="${EIDOS_BASE_PYTHONPATH:-}"
    export LD_LIBRARY_PATH="${EIDOS_BASE_LD_LIBRARY_PATH:-}"
    if [ -n "${EIDOS_BASE_VIRTUAL_ENV:-}" ]; then
        export VIRTUAL_ENV="$EIDOS_BASE_VIRTUAL_ENV"
    else
        unset VIRTUAL_ENV
    fi
}

eidos_use_env() {
    _eidos_capture_shell_baseline

    local path_value="${EIDOS_BASE_PATH:-${PATH:-}}"
    local pythonpath_value="${EIDOS_BASE_PYTHONPATH:-}"
    local ld_library_value="${EIDOS_BASE_LD_LIBRARY_PATH:-}"
    local forge_dir=""
    local bin_dir=""

    if [ -x "$VENV_PATH/bin/python" ]; then
        path_value="$(_eidos_path_prepend_unique "$path_value" "$VENV_PATH/bin")"
        export VIRTUAL_ENV="$VENV_PATH"
    fi

    for forge_dir in "$FORGE_ROOT"/*_forge "$FORGE_ROOT"/lib "$FORGE_ROOT"/eidos_mcp; do
        [ -d "$forge_dir" ] || continue
        bin_dir="$forge_dir/bin"
        if [ -d "$bin_dir" ]; then
            path_value="$(_eidos_path_prepend_unique "$path_value" "$bin_dir")"
        fi
        if [ -d "$forge_dir/src" ] && ! _eidos_path_contains "$pythonpath_value" "$forge_dir/src"; then
            pythonpath_value="${forge_dir}/src${pythonpath_value:+:$pythonpath_value}"
        fi
        if ! _eidos_path_contains "$pythonpath_value" "$forge_dir"; then
            pythonpath_value="${forge_dir}${pythonpath_value:+:$pythonpath_value}"
        fi
    done

    path_value="$(_eidos_path_prepend_unique "$path_value" "$FORGE_ROOT/bin")"
    if [ -d "$FORGE_ROOT/llama.cpp/build/bin" ]; then
        path_value="$(_eidos_path_prepend_unique "$path_value" "$FORGE_ROOT/llama.cpp/build/bin")"
        ld_library_value="$(_eidos_path_prepend_unique "$ld_library_value" "$FORGE_ROOT/llama.cpp/build/bin")"
    fi
    if [ -d "$FORGE_ROOT/llm_forge/bin" ]; then
        path_value="$(_eidos_path_prepend_unique "$path_value" "$FORGE_ROOT/llm_forge/bin")"
    fi

    export PATH="$path_value"
    export PYTHONPATH="$pythonpath_value"
    export LD_LIBRARY_PATH="$ld_library_value"
}

eidos_safe_exec() (
    eidos_use_env
    export PYTHONNOUSERSITE=1
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export TMPDIR="${TMPDIR:-${PREFIX:-/tmp}/tmp}"
    exec "$@"
)
