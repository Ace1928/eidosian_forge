#!/bin/bash
# Bash completion for metadata-forge CLI
# Auto-generated for Eidosian Forge

_metadata_forge_completion() {
    local cur prev words cword
    _init_completion || return

    local commands="template validate version"

    case "${prev}" in
        metadata-forge)
            COMPREPLY=($(compgen -W "${commands} --help -h" -- "${cur}"))
            return 0
            ;;
        template)
            COMPREPLY=($(compgen -W "-o --output --help" -- "${cur}"))
            return 0
            ;;
        validate)
            COMPREPLY=($(compgen -f -X '!*.@(json|yaml|yml)' -- "${cur}"))
            return 0
            ;;
        -o|--output)
            COMPREPLY=($(compgen -f -- "${cur}"))
            return 0
            ;;
    esac

    # Default: show commands
    if [[ ${cword} -eq 1 ]]; then
        COMPREPLY=($(compgen -W "${commands}" -- "${cur}"))
    fi
}

complete -F _metadata_forge_completion metadata-forge
