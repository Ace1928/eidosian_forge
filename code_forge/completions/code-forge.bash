# Bash completion for code-forge
# Add to ~/.bashrc: source <(path/to/code-forge --completion)

_code_forge_completions() {
    local cur prev commands
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    commands="status info integrations analyze index search ingest library stats"
    
    case "${COMP_CWORD}" in
        1)
            COMPREPLY=($(compgen -W "$commands --help --version --json --no-color --quiet" -- "$cur"))
            ;;
        *)
            COMPREPLY=()
            ;;
    esac
}

complete -F _code_forge_completions code-forge

