#!/usr/bin/env bash
set -euo pipefail

FORGE_ROOT="${EIDOS_FORGE_ROOT:-/data/data/com.termux/files/home/eidosian_forge}"
VENV_PATH="${FORGE_ROOT}/eidosian_venv"
BIN_DIR="${VENV_PATH}/bin"

mkdir -p "${BIN_DIR}"

if [ ! -e "${BIN_DIR}/pip" ] && [ -x "${BIN_DIR}/pip3.13" ]; then
  ln -sf "${BIN_DIR}/pip3.13" "${BIN_DIR}/pip"
fi

cat > "${BIN_DIR}/activate" <<EOF
# This file must be used with "source ${VENV_PATH}/bin/activate"

deactivate () {
    if [ -n "\${_OLD_VIRTUAL_PATH:-}" ]; then
        PATH="\${_OLD_VIRTUAL_PATH}"
        export PATH
        unset _OLD_VIRTUAL_PATH
    fi
    if [ -n "\${_OLD_VIRTUAL_PYTHONPATH:-}" ]; then
        PYTHONPATH="\${_OLD_VIRTUAL_PYTHONPATH}"
        export PYTHONPATH
        unset _OLD_VIRTUAL_PYTHONPATH
    else
        unset PYTHONPATH
    fi
    unset VIRTUAL_ENV
    unset VIRTUAL_ENV_PROMPT
    if [ -n "\${_OLD_PS1:-}" ]; then
        PS1="\${_OLD_PS1}"
        export PS1
        unset _OLD_PS1
    fi
    unset -f deactivate
}

_OLD_VIRTUAL_PATH="\${PATH:-}"
_OLD_VIRTUAL_PYTHONPATH="\${PYTHONPATH:-}"
export VIRTUAL_ENV="${VENV_PATH}"
export VIRTUAL_ENV_PROMPT="(eidosian_venv) "
PATH="${BIN_DIR}:\${PATH:-}"
export PATH
if [ -n "\${PS1:-}" ]; then
    _OLD_PS1="\${PS1:-}"
    PS1="\${VIRTUAL_ENV_PROMPT}\${PS1:-}"
    export PS1
fi
EOF

cat > "${BIN_DIR}/activate.fish" <<EOF
set -gx VIRTUAL_ENV "${VENV_PATH}"
set -gx VIRTUAL_ENV_PROMPT "(eidosian_venv) "
set -gx PATH "${BIN_DIR}" \$PATH
EOF

cat > "${BIN_DIR}/activate.csh" <<EOF
setenv VIRTUAL_ENV "${VENV_PATH}"
setenv VIRTUAL_ENV_PROMPT "(eidosian_venv) "
setenv PATH "${BIN_DIR}":\$PATH
EOF

chmod 644 "${BIN_DIR}/activate" "${BIN_DIR}/activate.fish" "${BIN_DIR}/activate.csh"
printf '%s\n' "${BIN_DIR}/activate"
