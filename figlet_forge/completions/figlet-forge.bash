#!/bin/bash
# Figlet Forge bash completions

_figlet_forge_completions() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    case "$prev" in
        figlet|figlet-forge)
            COMPREPLY=($(compgen -W "--font --list-fonts --width --justify --direction --reverse --flip --border --shade --color --color-list --unicode --output --html --svg --showcase --sample-text --sample-color --sample-fonts --version --help" -- "$cur"))
            ;;
        --font|-f)
            # Common fonts
            COMPREPLY=($(compgen -W "standard banner big block bubble digital ivrit lean mini script shadow slant small smslant" -- "$cur"))
            ;;
        --justify|-j)
            COMPREPLY=($(compgen -W "left right center auto" -- "$cur"))
            ;;
        --direction)
            COMPREPLY=($(compgen -W "auto left-to-right right-to-left" -- "$cur"))
            ;;
        --border)
            COMPREPLY=($(compgen -W "single double rounded bold shadow ascii" -- "$cur"))
            ;;
    esac
}

complete -F _figlet_forge_completions figlet
complete -F _figlet_forge_completions figlet-forge
