import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, List, Optional
import click
import typer
import typer.core
from click import Command, Group, Option
from . import __version__
def get_typer_from_state() -> Optional[typer.Typer]:
    spec = None
    if state.file:
        module_name = state.file.name
        spec = importlib.util.spec_from_file_location(module_name, str(state.file))
    elif state.module:
        spec = importlib.util.find_spec(state.module)
    if spec is None:
        if state.file:
            typer.echo(f'Could not import as Python file: {state.file}', err=True)
        else:
            typer.echo(f'Could not import as Python module: {state.module}', err=True)
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    obj = get_typer_from_module(module)
    return obj