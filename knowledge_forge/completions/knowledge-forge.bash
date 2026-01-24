# Bash completion for knowledge-forge
# Add to ~/.bashrc: source <(path/to/knowledge-forge --completion)

_knowledge_forge_completions() {
    local cur prev commands
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    commands="status info integrations list search add link path concepts unified stats delete"
    
    case "${COMP_CWORD}" in
        1)
            COMPREPLY=($(compgen -W "$commands --help --version --json --no-color --quiet" -- "$cur"))
            ;;
        *)
            COMPREPLY=()
            ;;
    esac
}

complete -F _knowledge_forge_completions knowledge-forge

