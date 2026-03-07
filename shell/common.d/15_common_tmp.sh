#!/usr/bin/env bash

eidos_shell_common_tmp_init() {
    eidos_shell_is_interactive || return 0

    local default_tmp=""
    local target_link="${HOME}/tmp"
    local forge_root="${EIDOS_FORGE_ROOT:-${HOME}/eidosian_forge}"
    local preload_lib="${EIDOS_TMP_PRELOAD_LIB:-${forge_root}/build/libeidos_tmpredir.so}"

    if [ -n "${EIDOS_TMPDIR:-}" ]; then
        default_tmp="${EIDOS_TMPDIR}"
    elif eidos_shell_is_termux; then
        default_tmp="${PREFIX:-/data/data/com.termux/files/usr}/tmp/eidos-${USER:-termux}"
    else
        default_tmp="${HOME}/tmp"
    fi

    export EIDOS_TMPDIR="${default_tmp}"
    export TMPDIR="${EIDOS_TMPDIR}"
    export TMP="${EIDOS_TMPDIR}"
    export TEMP="${EIDOS_TMPDIR}"
    mkdir -p "${EIDOS_TMPDIR}"

    if [ "${target_link}" != "${EIDOS_TMPDIR}" ]; then
        if [ -L "${target_link}" ]; then
            local linked
            linked="$(readlink "${target_link}" 2>/dev/null || true)"
            if [ "${linked}" != "${EIDOS_TMPDIR}" ]; then
                rm -f "${target_link}"
                ln -s "${EIDOS_TMPDIR}" "${target_link}" || true
            fi
        elif [ ! -e "${target_link}" ]; then
            ln -s "${EIDOS_TMPDIR}" "${target_link}" || true
        fi
    fi

    if [ "${EIDOS_ENABLE_TMP_PRELOAD:-0}" = "1" ] && [ -f "${preload_lib}" ]; then
        case ":${LD_PRELOAD:-}:" in
            *":${preload_lib}:"*) ;;
            *) export LD_PRELOAD="${preload_lib}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
        esac
    fi
}
