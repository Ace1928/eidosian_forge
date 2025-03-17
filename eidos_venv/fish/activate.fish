function _eidos_venv_prompt
    echo -n "🔮 (eidos) "
end

if status is-interactive
    set -g __eidos_venv_old_prompt $fish_prompt
    # Save current Python path to restore later
    set -g __eidos_venv_old_pythonpath $PYTHONPATH
    
    # Add project directories to PYTHONPATH
    for repo in repo_forge_repo metadata_forge_repo ollama_forge_repo terminal_forge_repo doc_forge_repo memory_forge_repo refactor_forge_repo file_forge_repo knowledge_forge_repo
        set repo_path "/home/lloyd/repos/$repo"
        if test -d $repo_path
            set -x PYTHONPATH $repo_path $PYTHONPATH
        end
    end
    
    # Set fancy prompt
    functions -c fish_prompt __eidos_venv_old_fish_prompt
    function fish_prompt
        _eidos_venv_prompt
        __eidos_venv_old_fish_prompt
    end
end

function deactivate
    # Restore old environment
    if functions -q __eidos_venv_old_fish_prompt
        functions -e fish_prompt
        functions -c __eidos_venv_old_fish_prompt fish_prompt
        functions -e __eidos_venv_old_fish_prompt
    end
    
    # Restore old Python path
    if set -q __eidos_venv_old_pythonpath
        set -x PYTHONPATH $__eidos_venv_old_pythonpath
        set -e __eidos_venv_old_pythonpath
    end
    
    # Print exit message
    echo "Deactivated Eidosian virtual environment"
end
