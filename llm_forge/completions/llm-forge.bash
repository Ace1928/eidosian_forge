# Bash completion for llm-forge
# Add to ~/.bashrc: source <(path/to/llm-forge --completion)

_llm_forge_completions() {
    local cur prev commands
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    commands="status info integrations models chat embed config test"
    
    case "${COMP_CWORD}" in
        1)
            COMPREPLY=($(compgen -W "$commands --help --version --json --no-color --quiet" -- "$cur"))
            ;;
        *)
            COMPREPLY=()
            ;;
    esac
}

complete -F _llm_forge_completions llm-forge

