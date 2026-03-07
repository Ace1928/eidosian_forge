#!/usr/bin/env bash

eidos_termux_runtime_init() {
    eidos_termux_is_interactive || return 0

    export TU_DEBUG="${TU_DEBUG:-noconform}"
    export MESA_VK_WSI_PRESENT_MODE="${MESA_VK_WSI_PRESENT_MODE:-fifo}"
    export MESA_VK_ABORT_ON_DEVICE_LOSS="${MESA_VK_ABORT_ON_DEVICE_LOSS:-true}"
    export MESA_SHADER_CACHE_DISABLE="${MESA_SHADER_CACHE_DISABLE:-false}"
    export MESA_NO_DITHER="${MESA_NO_DITHER:-1}"
    export BLIS_ARCH="${BLIS_ARCH:-generic}"
    export DISPLAY="${DISPLAY:-:1}"

    local runtime_default="/data/data/com.termux/files/usr/tmp/$(id -u)"
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${runtime_default}}"
    mkdir -p "${XDG_RUNTIME_DIR}"
    chmod 700 "${XDG_RUNTIME_DIR}" >/dev/null 2>&1 || true

    local lib_flags='-lm -ldl -llog -lGLESv1_CM -lGLESv2 -landroid'
    export EXTRA_LDFLAGS="${EXTRA_LDFLAGS:+$EXTRA_LDFLAGS }${lib_flags}"
    export LDFLAGS="${LDFLAGS:+$LDFLAGS }${lib_flags}"
    export LIBS="${LIBS:+$LIBS }${lib_flags}"
    export LIBRARIES="${lib_flags}"

    alias docs='cd ~/storage/shared/Documents'
    alias dls='cd ~/storage/shared/Download'
    alias pics='cd ~/storage/shared/DCIM'

    if [ "${EIDOS_ENABLE_PULSEAUDIO_AUTOSTART:-1}" = "1" ] && eidos_shell_has pulseaudio; then
        if ! pgrep -x pulseaudio >/dev/null 2>&1; then
            pulseaudio --start --exit-idle-time=-1 >/dev/null 2>&1 || true
            sleep 1
            if eidos_shell_has pactl; then
                pactl load-module module-native-protocol-tcp auth-anonymous=1 listen=127.0.0.1 >/dev/null 2>&1 || true
            fi
        fi
    fi

    if [ "${EIDOS_ENABLE_X11_AUTOSTART:-1}" = "1" ] && [ -x "${HOME}/scripts/start_x11" ]; then
        if ! pgrep -f "${HOME}/scripts/start_x11" >/dev/null 2>&1; then
            nohup "${HOME}/scripts/start_x11" >/dev/null 2>&1 &
            disown || true
        fi
    fi

    if [ "${EIDOS_DISABLE_NOTIFICATIONS:-0}" != "1" ]; then
        notify 'Shell loaded and ready.' >/dev/null 2>&1 || true
    fi
}
