import os
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import click
def install_fish(*, prog_name: str, complete_var: str, shell: str) -> Path:
    path_obj = Path.home() / f'.config/fish/completions/{prog_name}.fish'
    parent_dir: Path = path_obj.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    script_content = get_completion_script(prog_name=prog_name, complete_var=complete_var, shell=shell)
    path_obj.write_text(f'{script_content}\n')
    return path_obj