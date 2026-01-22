import os
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import click
def install_zsh(*, prog_name: str, complete_var: str, shell: str) -> Path:
    zshrc_path = Path.home() / '.zshrc'
    zshrc_path.parent.mkdir(parents=True, exist_ok=True)
    zshrc_content = ''
    if zshrc_path.is_file():
        zshrc_content = zshrc_path.read_text()
    completion_init_lines = ['autoload -Uz compinit', 'compinit', "zstyle ':completion:*' menu select", 'fpath+=~/.zfunc']
    for line in completion_init_lines:
        if line not in zshrc_content:
            zshrc_content += f'\n{line}'
    zshrc_content += '\n'
    zshrc_path.write_text(zshrc_content)
    path_obj = Path.home() / f'.zfunc/_{prog_name}'
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    script_content = get_completion_script(prog_name=prog_name, complete_var=complete_var, shell=shell)
    path_obj.write_text(script_content)
    return path_obj