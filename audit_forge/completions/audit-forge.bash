#!/bin/bash
# Bash completion for audit-forge CLI
# Auto-generated for Eidosian Forge

_audit_forge_completion() {
    local cur prev words cword
    _init_completion || return

    local commands="coverage mark todo"

    case "${prev}" in
        audit-forge)
            COMPREPLY=($(compgen -W "${commands} --help" -- "${cur}"))
            return 0
            ;;
        coverage)
            # Suggest directories
            COMPREPLY=($(compgen -d -- "${cur}"))
            return 0
            ;;
        mark)
            # Suggest files
            COMPREPLY=($(compgen -f -- "${cur}"))
            return 0
            ;;
        --agent)
            COMPREPLY=($(compgen -W "user eidos agent" -- "${cur}"))
            return 0
            ;;
        --section)
            COMPREPLY=($(compgen -W "Immediate Short-Term Long-Term" -- "${cur}"))
            return 0
            ;;
    esac

    # Default: show main commands
    if [[ ${cword} -eq 1 ]]; then
        COMPREPLY=($(compgen -W "${commands}" -- "${cur}"))
    fi
}

complete -F _audit_forge_completion audit-forge
