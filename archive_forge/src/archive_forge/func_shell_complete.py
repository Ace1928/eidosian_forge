import os
import sys
from typing import Any, MutableMapping, Tuple
import click
from ._completion_classes import completion_init
from ._completion_shared import Shells, get_completion_script, install
from .models import ParamMeta
from .params import Option
from .utils import get_params_from_function
def shell_complete(cli: click.Command, ctx_args: MutableMapping[str, Any], prog_name: str, complete_var: str, instruction: str) -> int:
    import click
    import click.shell_completion
    if '_' not in instruction:
        click.echo('Invalid completion instruction.', err=True)
        return 1
    instruction, _, shell = instruction.partition('_')
    comp_cls = click.shell_completion.get_completion_class(shell)
    if comp_cls is None:
        click.echo(f'Shell {shell} not supported.', err=True)
        return 1
    comp = comp_cls(cli, ctx_args, prog_name, complete_var)
    if instruction == 'source':
        click.echo(comp.source())
        return 0
    if instruction == 'complete':
        click.echo(comp.complete())
        return 0
    click.echo(f'Completion instruction "{instruction}" not supported.', err=True)
    return 1