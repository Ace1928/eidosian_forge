#!/usr/bin/env bash
# Eidosian shell bootstrap wrapper.
EIDOS_SHELL_BOOTSTRAP="${EIDOS_SHELL_BOOTSTRAP:-$HOME/eidosian_forge/shell/bootstrap.sh}"
if [ -f "${EIDOS_SHELL_BOOTSTRAP}" ]; then
    # shellcheck source=/dev/null
    source "${EIDOS_SHELL_BOOTSTRAP}"
else
    printf '[ERROR] Missing Eidosian shell bootstrap at %s\n' "${EIDOS_SHELL_BOOTSTRAP}" >&2
fi
