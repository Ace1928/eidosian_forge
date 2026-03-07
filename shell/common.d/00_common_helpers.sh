#!/usr/bin/env bash

if [ -z "${EIDOS_SHELL_COMMON_HELPERS_LOADED:-}" ]; then
    export EIDOS_SHELL_COMMON_HELPERS_LOADED=1
fi

DEBUG="${DEBUG:-0}"

log_debug() {
    [ "${DEBUG:-0}" = "1" ] || return 0
    printf '[\033[34mDEBUG\033[0m] %s\n' "$*"
}

log_info() {
    printf '[\033[32mINFO\033[0m] %s\n' "$*"
}

log_warn() {
    printf '[\033[33mWARN\033[0m] %s\n' "$*" >&2
}

log_error() {
    printf '[\033[31mERROR\033[0m] %s\n' "$*" >&2
}

eidos_shell_log() {
    local level="${1:-INFO}"
    shift || true
    case "${level}" in
        DEBUG) log_debug "$*" ;;
        WARN) log_warn "$*" ;;
        ERROR) log_error "$*" ;;
        *) log_info "$*" ;;
    esac
}

eidos_shell_is_interactive() {
    case "$-" in
        *i*) return 0 ;;
        *) return 1 ;;
    esac
}

eidos_shell_is_termux() {
    case "${PREFIX:-}" in
        *com.termux*) return 0 ;;
    esac
    [ -n "${TERMUX_VERSION:-}" ]
}

eidos_shell_is_linux() {
    [ "$(uname -s 2>/dev/null || printf unknown)" = "Linux" ]
}

eidos_shell_has() {
    command -v "$1" >/dev/null 2>&1
}

eidos_path_contains() {
    case ":${1:-}:" in
        *":${2:-}:"*) return 0 ;;
        *) return 1 ;;
    esac
}

eidos_path_prepend() {
    local entry="${1:-}"
    [ -n "${entry}" ] || return 0
    if [ -d "${entry}" ] && ! eidos_path_contains "${PATH:-}" "${entry}"; then
        PATH="${entry}${PATH:+:${PATH}}"
    fi
}

eidos_path_append() {
    local entry="${1:-}"
    [ -n "${entry}" ] || return 0
    if [ -d "${entry}" ] && ! eidos_path_contains "${PATH:-}" "${entry}"; then
        PATH="${PATH:+${PATH}:}${entry}"
    fi
}

notify() {
    if eidos_shell_has termux-notification; then
        termux-notification --title "Eidosian Shell" --content "$*" --priority high --id 99 >/dev/null 2>&1 &
    else
        printf 'Notification: %s\n' "$*"
    fi
}

update_title() {
    printf '\033]0;%s\007' "${PWD}"
}

mkcd() {
    mkdir -p -- "$1" && cd -P -- "$1"
}

extract() {
    if [ ! -f "${1:-}" ]; then
        printf "'%s' is not a valid file\n" "$1"
        return 1
    fi
    case "$1" in
        *.tar.bz2) tar xjf "$1" ;;
        *.tar.gz) tar xzf "$1" ;;
        *.tar.xz) tar xJf "$1" ;;
        *.bz2) bunzip2 "$1" ;;
        *.rar) unrar x "$1" ;;
        *.gz) gunzip "$1" ;;
        *.tar) tar xf "$1" ;;
        *.tbz2) tar xjf "$1" ;;
        *.tgz) tar xzf "$1" ;;
        *.zip) unzip "$1" ;;
        *.Z) uncompress "$1" ;;
        *.7z) 7z x "$1" ;;
        *) printf "extract: '%s' cannot be extracted via extract()\n" "$1" ; return 1 ;;
    esac
}

sysinfo() {
    printf 'Uptime: %s\n' "$(uptime 2>/dev/null || printf unavailable)"
    if eidos_shell_has free; then
        free -h
    else
        printf 'free command not available\n'
    fi
}

list_aliases() {
    alias -p
}

eidos_trim() {
    local value="${1:-}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

eidos_load_dotenv() {
    local file_path="${1:-.env}"
    local line key value
    [ -f "${file_path}" ] || return 0

    while IFS= read -r line || [ -n "${line}" ]; do
        line="$(eidos_trim "${line}")"
        case "${line}" in
            ""|\#*) continue ;;
            export\ *) line="${line#export }" ;;
        esac
        case "${line}" in
            *=*) ;;
            *) continue ;;
        esac

        key="$(eidos_trim "${line%%=*}")"
        value="$(eidos_trim "${line#*=}")"
        [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

        case "${value}" in
            \"*\") value="${value#\"}"; value="${value%\"}" ;;
            \'*\') value="${value#\'}"; value="${value%\'}" ;;
        esac

        printf -v "${key}" '%s' "${value}"
        export "${key}"
    done < "${file_path}"
}
