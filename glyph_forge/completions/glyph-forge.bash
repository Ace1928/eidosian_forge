#!/bin/bash
# Bash completion for glyph-forge CLI
# Auto-generated for Eidosian Forge

_glyph_forge_completion() {
    local cur prev words cword
    _init_completion || return

    # Main commands
    local commands="version interactive list-commands bannerize imagize"
    
    # Bannerize subcommands
    local bannerize_commands="text"
    
    # Imagize subcommands
    local imagize_commands="convert"

    case "${prev}" in
        glyph-forge)
            COMPREPLY=($(compgen -W "${commands} --help -h" -- "${cur}"))
            return 0
            ;;
        bannerize)
            COMPREPLY=($(compgen -W "${bannerize_commands} --help -h" -- "${cur}"))
            return 0
            ;;
        imagize)
            COMPREPLY=($(compgen -W "${imagize_commands} --help -h" -- "${cur}"))
            return 0
            ;;
        convert)
            # Suggest image files
            COMPREPLY=($(compgen -f -X '!*.@(png|jpg|jpeg|gif|bmp|webp)' -- "${cur}"))
            return 0
            ;;
        --font|-f)
            # Font options
            COMPREPLY=($(compgen -W "slant banner small standard big" -- "${cur}"))
            return 0
            ;;
        --width|-w)
            COMPREPLY=($(compgen -W "40 60 80 100 120 160 200" -- "${cur}"))
            return 0
            ;;
        --charset|-c)
            COMPREPLY=($(compgen -W "general simple blocks detailed" -- "${cur}"))
            return 0
            ;;
        --output|-o)
            COMPREPLY=($(compgen -f -- "${cur}"))
            return 0
            ;;
    esac

    # Default: show main commands
    if [[ ${cword} -eq 1 ]]; then
        COMPREPLY=($(compgen -W "${commands}" -- "${cur}"))
    fi
}

complete -F _glyph_forge_completion glyph-forge
