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
@cmd.command('config')
def run_update_build_config(path: Path=typer.Argument(BUILD_CONFIG_PATH, help='Path to Build Config', resolve_path=True)):
    """
    Runs the build config
    """
    if not path.exists():
        echo(f'{COLOR.RED}File does not exist: {path.as_posix()}{COLOR.END}')
        raise ValueError(f'File does not exist: {path.as_posix()}')
    echo(f'{COLOR.BLUE}Updating Build Config: {path.as_posix()}{COLOR.END}')
    config.update_config(path)