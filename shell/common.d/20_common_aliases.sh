#!/usr/bin/env bash

eidos_shell_common_aliases_init() {
    eidos_shell_is_interactive || return 0

    alias sysinfo='sysinfo'
    alias aliases='list_aliases'
    alias nb-serve='nb browse --serve'
    if eidos_shell_has eza; then
        alias ls='eza -lF --icons'
    fi
    if eidos_shell_has bat; then
        alias cat='bat'
    fi
    alias ll='ls -alF'
    alias la='ls -A'
    alias l='ls -CF'
    alias grep='grep --color=auto'
    alias ..='cd ..'
    alias ...='cd ../..'
    alias cls='clear'
    alias h='history'
    alias vi='vim'
    alias c='clear'

    [ -x "$HOME/scripts/display_os_info" ] && alias osinfo='$HOME/scripts/display_os_info'
    [ -f "$HOME/scripts/aiml_launcher.py" ] && alias aiml='$HOME/scripts/aiml_launcher.py'
    [ -f "$HOME/scripts/system_dashboard.py" ] && alias sysdash='$HOME/scripts/system_dashboard.py'
    [ -x "$HOME/scripts/update_packages.sh" ] && alias pkgup='$HOME/scripts/update_packages.sh'
    [ -f "$HOME/scripts/syscheck.py" ] && alias syscheck='python3 $HOME/scripts/syscheck.py'
    [ -x "$HOME/scripts/backup.sh" ] && alias backup='$HOME/scripts/backup.sh'
}
