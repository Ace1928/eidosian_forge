import os
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import click
def get_completion_script(*, prog_name: str, complete_var: str, shell: str) -> str:
    cf_name = _invalid_ident_char_re.sub('', prog_name.replace('-', '_'))
    script = _completion_scripts.get(shell)
    if script is None:
        click.echo(f'Shell {shell} not supported.', err=True)
        raise click.exceptions.Exit(1)
    return (script % {'complete_func': f'_{cf_name}_completion', 'prog_name': prog_name, 'autocomplete_var': complete_var}).strip()