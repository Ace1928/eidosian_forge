#!/usr/bin/env bash

EIDOS_SHELL_ROOT="${EIDOS_SHELL_ROOT:-/data/data/com.termux/files/home/eidosian_forge/shell}"
EIDOS_SHELL_COMMON_DIR="${EIDOS_SHELL_COMMON_DIR:-${EIDOS_SHELL_ROOT}/common.d}"

for eidos_common_module in \
    "${EIDOS_SHELL_COMMON_DIR}/00_common_helpers.sh" \
    "${EIDOS_SHELL_COMMON_DIR}/10_common_runtime.sh" \
    "${EIDOS_SHELL_COMMON_DIR}/20_common_aliases.sh" \
    "${EIDOS_SHELL_COMMON_DIR}/30_common_prompt.sh"
do
    [ -f "${eidos_common_module}" ] || continue
    # shellcheck source=/dev/null
    source "${eidos_common_module}"
done
unset eidos_common_module

eidos_shell_common_runtime_init
eidos_shell_common_aliases_init
eidos_shell_common_prompt_init

if eidos_shell_is_termux && [ -f "${EIDOS_SHELL_ROOT}/termux_bootstrap.sh" ]; then
    # shellcheck source=/dev/null
    source "${EIDOS_SHELL_ROOT}/termux_bootstrap.sh"
fi
