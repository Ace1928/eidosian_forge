#!/usr/bin/env bash

eidos_shell_common_prompt_init() {
    eidos_shell_is_interactive || return 0

    if eidos_shell_has starship; then
        eval "$(starship init bash)"
    elif [ -z "${EIDOS_SIMPLE_PS1_SET:-}" ]; then
        export EIDOS_SIMPLE_PS1_SET=1
        PS1='\u@\h:\w\\$ '
    fi

    if [ -x "$(command -v command-not-found 2>/dev/null || true)" ]; then
        command_not_found_handle() {
            command-not-found -- "$1"
            return 127
        }
    fi
}
