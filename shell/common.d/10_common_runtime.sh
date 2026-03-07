#!/usr/bin/env bash

eidos_shell_common_runtime_init() {
    eidos_shell_is_interactive || return 0

    export LANG="${LANG:-en_US.UTF-8}"
    export TERM="${TERM:-xterm-256color}"
    export SCRIPTS_DIR="${SCRIPTS_DIR:-$HOME/scripts}"
    export NB_PORT="${NB_PORT:-9090}"
    export USE_CUDA="${USE_CUDA:-0}"
    export AUDIT_MAX_MB="${AUDIT_MAX_MB:-5}"

    umask 022
    shopt -s checkwinsize autocd extglob globstar histappend
    set -o noclobber

    export HISTSIZE="${HISTSIZE:-10000}"
    export HISTFILESIZE="${HISTFILESIZE:-20000}"
    export HISTCONTROL="${HISTCONTROL:-ignoreboth:erasedups}"

    PROMPT_COMMAND="update_title${PROMPT_COMMAND:+;$PROMPT_COMMAND}"

    eidos_path_prepend "$HOME/bin"
    eidos_path_prepend "$HOME/.local/bin"
    eidos_path_prepend "$HOME/.cargo/bin"
    eidos_path_append "/data/data/com.termux/files/usr/local/bin"
    eidos_path_append "/usr/local/bin"

    if [ -n "${GOPATH:-}" ]; then
        eidos_path_append "${GOPATH}/bin"
    elif [ -d "$HOME/go/bin" ]; then
        eidos_path_append "$HOME/go/bin"
    fi

    if [ -d "$HOME/.local/lib" ]; then
        export LD_LIBRARY_PATH="$HOME/.local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi

    [ -f "$HOME/.autojump/etc/profile.d/autojump.sh" ] && . "$HOME/.autojump/etc/profile.d/autojump.sh"
    eidos_load_dotenv .env
}
