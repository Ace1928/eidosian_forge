# Bash completion for memory-forge
# Add to ~/.bashrc: source <(path/to/memory-forge --completion)

_memory_forge_completions() {
    local cur prev commands
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    commands="status info integrations list search store introspect context stats cleanup"
    
    case "${COMP_CWORD}" in
        1)
            COMPREPLY=($(compgen -W "$commands --help --version --json --no-color --quiet" -- "$cur"))
            ;;
        *)
            COMPREPLY=()
            ;;
    esac
}

complete -F _memory_forge_completions memory-forge

