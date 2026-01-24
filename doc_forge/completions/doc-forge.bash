#!/bin/bash
# Doc Forge bash completions

_doc_forge_completions() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    case "$prev" in
        doc|doc-forge)
            COMPREPLY=($(compgen -W "build setup clean check serve --debug --version --help" -- "$cur"))
            ;;
        build)
            COMPREPLY=($(compgen -W "--format --no-fix --help" -- "$cur"))
            ;;
        --format)
            COMPREPLY=($(compgen -W "html pdf epub" -- "$cur"))
            ;;
        serve)
            COMPREPLY=($(compgen -W "--port --help" -- "$cur"))
            ;;
    esac
}

complete -F _doc_forge_completions doc
complete -F _doc_forge_completions doc-forge
