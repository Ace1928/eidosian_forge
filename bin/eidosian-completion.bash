#!/bin/bash
# Eidosian Forge bash completions
# Source this file: source /path/to/eidosian_forge/bin/eidosian-completion.bash

# Main eidosian command completion
_eidosian_completions() {
    local cur prev commands forges
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    commands="status forges install memory knowledge code llm word crawl glyph audit refactor metadata terminal agent agent-daemon agent-top doc figlet game repo control version type file diagnostics narrative ollama gis article viz test mkey sms prompt lyrics erais moltbook web"
    
    case "${COMP_CWORD}" in
        1)
            COMPREPLY=($(compgen -W "$commands --help --version --json --no-color" -- "$cur"))
            ;;
        2)
            case "$prev" in
                memory)
                    COMPREPLY=($(compgen -W "status list search store introspect context stats cleanup --help" -- "$cur"))
                    ;;
                knowledge)
                    COMPREPLY=($(compgen -W "status list search add link path concepts unified stats delete --help" -- "$cur"))
                    ;;
                code)
                    COMPREPLY=($(compgen -W "status analyze index search ingest library stats --help" -- "$cur"))
                    ;;
                llm)
                    COMPREPLY=($(compgen -W "status models chat embed config test --help" -- "$cur"))
                    ;;
                word)
                    COMPREPLY=($(compgen -W "status lookup define related synsets graph build --help" -- "$cur"))
                    ;;
                crawl)
                    COMPREPLY=($(compgen -W "status fetch extract robots tika cache --help" -- "$cur"))
                    ;;
                glyph)
                    COMPREPLY=($(compgen -W "version interactive list-commands bannerize imagize --help" -- "$cur"))
                    ;;
                audit)
                    COMPREPLY=($(compgen -W "coverage mark todo --help" -- "$cur"))
                    ;;
                refactor)
                    COMPREPLY=($(compgen -W "--analyze-only --dry-run -o --output-dir -n --package-name -v --verbose --clean --help" -- "$cur"))
                    ;;
                metadata)
                    COMPREPLY=($(compgen -W "template validate version --help" -- "$cur"))
                    ;;
                terminal)
                    COMPREPLY=($(compgen -W "status colors themes banner ascii layout --help" -- "$cur"))
                    ;;
                agent)
                    COMPREPLY=($(compgen -W "state journal goals steps runs --help" -- "$cur"))
                    ;;
                agent-daemon)
                    COMPREPLY=($(compgen -W "--once --loop --tick --jitter --help" -- "$cur"))
                    ;;
                agent-top)
                    COMPREPLY=($(compgen -W "--help" -- "$cur"))
                    ;;
                doc)
                    COMPREPLY=($(compgen -W "build setup clean check serve --help" -- "$cur"))
                    ;;
                figlet)
                    COMPREPLY=($(compgen -W "--font --list-fonts --width --justify --color --showcase --help" -- "$cur"))
                    ;;
                game)
                    COMPREPLY=($(compgen -W "demo bench index ingest report --help" -- "$cur"))
                    ;;
                repo)
                    COMPREPLY=($(compgen -W "create verify --verbose --dry-run --help" -- "$cur"))
                    ;;
                control)
                    COMPREPLY=($(compgen -W "status safety mouse type screen daemon --help" -- "$cur"))
                    ;;
                version)
                    COMPREPLY=($(compgen -W "show bump compare check migrate --help" -- "$cur"))
                    ;;
                type)
                    COMPREPLY=($(compgen -W "register validate list status --help" -- "$cur"))
                    ;;
                file)
                    COMPREPLY=($(compgen -W "info hash tree status --help" -- "$cur"))
                    ;;
                diagnostics)
                    COMPREPLY=($(compgen -W "system python memory disk health status --help" -- "$cur"))
                    ;;
                narrative)
                    COMPREPLY=($(compgen -W "run config setup --help" -- "$cur"))
                    ;;
                ollama)
                    COMPREPLY=($(compgen -W "list chat generate status --help" -- "$cur"))
                    ;;
                gis)
                    COMPREPLY=($(compgen -W "distance geocode status --help" -- "$cur"))
                    ;;
                moltbook)
                    COMPREPLY=($(compgen -W "sanitize screen validate quarantine skill-review ingest bootstrap --list --help" -- "$cur"))
                    ;;
                article|viz|test|mkey|sms|prompt|lyrics|erais)
                    COMPREPLY=($(compgen -W "status --help" -- "$cur"))
                    ;;
                web)
                    COMPREPLY=($(compgen -W "serve status --help" -- "$cur"))
                    ;;
                install)
                    COMPREPLY=($(compgen -W "--completions" -- "$cur"))
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            ;;
        *)
            COMPREPLY=()
            ;;
    esac
}

complete -F _eidosian_completions eidosian

echo "Eidosian Forge completions loaded (35 forges)."
