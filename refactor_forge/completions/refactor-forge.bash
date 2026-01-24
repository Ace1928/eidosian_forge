#!/bin/bash
# Bash completion for refactor-forge CLI
# Auto-generated for Eidosian Forge

_refactor_forge_completion() {
    local cur prev words cword
    _init_completion || return

    case "${prev}" in
        refactor-forge)
            COMPREPLY=($(compgen -f -X '!*.py' -- "${cur}"))
            return 0
            ;;
        -o|--output-dir)
            COMPREPLY=($(compgen -d -- "${cur}"))
            return 0
            ;;
        -n|--package-name)
            return 0
            ;;
    esac

    # Options
    local opts="-h --help -o --output-dir -n --package-name --analyze-only --dry-run -v --verbose --clean --version"
    COMPREPLY=($(compgen -W "${opts}" -- "${cur}"))
}

complete -F _refactor_forge_completion refactor-forge
