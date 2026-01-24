#!/bin/bash
# Bash completion for terminal-forge CLI
# Auto-generated for Eidosian Forge

_terminal_forge_completion() {
    local cur prev words cword
    _init_completion || return

    local commands="status info integrations colors themes banner ascii layout"

    case "${prev}" in
        terminal-forge)
            COMPREPLY=($(compgen -W "${commands} --help -h --version -v" -- "${cur}"))
            return 0
            ;;
        colors)
            COMPREPLY=($(compgen -W "--show --json --help" -- "${cur}"))
            return 0
            ;;
        themes)
            COMPREPLY=($(compgen -W "--apply --json --help" -- "${cur}"))
            return 0
            ;;
        banner)
            COMPREPLY=($(compgen -W "--title --border --color --help" -- "${cur}"))
            return 0
            ;;
        ascii)
            COMPREPLY=($(compgen -f -X '!*.@(png|jpg|jpeg|gif|bmp|webp)' -- "${cur}"))
            return 0
            ;;
        --border)
            COMPREPLY=($(compgen -W "single double rounded bold none" -- "${cur}"))
            return 0
            ;;
        --charset)
            COMPREPLY=($(compgen -W "standard blocks braille detailed" -- "${cur}"))
            return 0
            ;;
        --width)
            COMPREPLY=($(compgen -W "40 60 80 100 120" -- "${cur}"))
            return 0
            ;;
    esac

    # Default: show commands
    if [[ ${cword} -eq 1 ]]; then
        COMPREPLY=($(compgen -W "${commands}" -- "${cur}"))
    fi
}

complete -F _terminal_forge_completion terminal-forge
