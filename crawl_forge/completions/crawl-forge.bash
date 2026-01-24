# Bash completion for crawl-forge
# Add to ~/.bashrc: source <(path/to/crawl-forge --completion)

_crawl_forge_completions() {
    local cur prev commands
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    commands="status info integrations fetch extract robots tika cache"
    
    case "${COMP_CWORD}" in
        1)
            COMPREPLY=($(compgen -W "$commands --help --version --json --no-color --quiet" -- "$cur"))
            ;;
        *)
            COMPREPLY=()
            ;;
    esac
}

complete -F _crawl_forge_completions crawl-forge

