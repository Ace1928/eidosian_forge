import os
import sys
from typing import Any, MutableMapping, Tuple
import click
from ._completion_classes import completion_init
from ._completion_shared import Shells, get_completion_script, install
from .models import ParamMeta
from .params import Option
from .utils import get_params_from_function
def install_callback(ctx: click.Context, param: click.Parameter, value: Any) -> Any:
    if not value or ctx.resilient_parsing:
        return value
    if isinstance(value, str):
        shell, path = install(shell=value)
    else:
        shell, path = install()
    click.secho(f'{shell} completion installed in {path}', fg='green')
    click.echo('Completion will take effect once you restart the terminal')
    sys.exit(0)