import os
import yaml
import typer
import contextlib
import subprocess
import concurrent.futures
from pathlib import Path
from pydantic import model_validator
from lazyops.types.models import BaseModel
from lazyops.libs.proxyobj import ProxyObject
from typing import Optional, List, Any, Dict, Union
@cmd.command('hotfix')
def run_pip_hotfix(file: Path=typer.Argument(..., help='Filename to Hotfix', resolve_path=True)):
    """
    Runs hotfix for pip requirements
    """
    if not file.exists():
        echo(f'{COLOR.RED}File does not exist: {file.as_posix()}{COLOR.END}')
        raise ValueError(f'File does not exist: {file.as_posix()}')
    echo(f'{COLOR.BLUE}Running Hotfix for {file.as_posix()}{COLOR.END}')
    reqs = parse_text_file(file)
    for req in reqs:
        echo(f'{COLOR.BLUE}Installing: {req}{COLOR.END}')
        if 'GITHUB_TOKEN' in req:
            req = req.replace('GITHUB_TOKEN', GITHUB_TOKEN)
        os.system(f'pip install --upgrade --no-deps --force-reinstall "{req}"')