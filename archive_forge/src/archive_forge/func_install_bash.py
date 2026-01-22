import os
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import click
def install_bash(*, prog_name: str, complete_var: str, shell: str) -> Path:
    completion_path = Path.home() / f'.bash_completions/{prog_name}.sh'
    rc_path = Path.home() / '.bashrc'
    rc_path.parent.mkdir(parents=True, exist_ok=True)
    rc_content = ''
    if rc_path.is_file():
        rc_content = rc_path.read_text()
    completion_init_lines = [f'source {completion_path}']
    for line in completion_init_lines:
        if line not in rc_content:
            rc_content += f'\n{line}'
    rc_content += '\n'
    rc_path.write_text(rc_content)
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    script_content = get_completion_script(prog_name=prog_name, complete_var=complete_var, shell=shell)
    completion_path.write_text(script_content)
    return completion_path