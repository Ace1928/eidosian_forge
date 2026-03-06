#!/usr/bin/env bash

eidos_termux_runtime_init() {
    eidos_termux_is_interactive || return 0

    export BLIS_ARCH="${BLIS_ARCH:-generic}"
    local lib_flags="-lm -ldl -llog -lGLESv1_CM -lGLESv2 -landroid"
    export EXTRA_LDFLAGS="${EXTRA_LDFLAGS:+$EXTRA_LDFLAGS }${lib_flags}"
    export LDFLAGS="${LDFLAGS:+$LDFLAGS }${lib_flags}"
    export LIBS="${LIBS:+$LIBS }${lib_flags}"
    export LIBRARIES="${lib_flags}"

    alias docs='cd ~/storage/shared/Documents'
    alias dls='cd ~/storage/shared/Download'
    alias pics='cd ~/storage/shared/DCIM'

    if [ -f .env ]; then
        # shellcheck source=/dev/null
        source .env
    fi

    if [ "${EIDOS_ENABLE_PULSEAUDIO_AUTOSTART:-1}" = "1" ] && command -v pulseaudio >/dev/null 2>&1; then
        if ! pgrep -x pulseaudio >/dev/null 2>&1; then
            pulseaudio --start --exit-idle-time=-1 >/dev/null 2>&1 || true
            sleep 1
            if command -v pactl >/dev/null 2>&1; then
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

    if [ "${EIDOS_DISABLE_NOTIFICATIONS:-0}" != "1" ] && command -v notify >/dev/null 2>&1; then
        notify "Shell loaded and ready."
    fi
}
