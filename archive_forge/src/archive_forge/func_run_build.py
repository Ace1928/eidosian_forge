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
@cmd.command('run')
def run_build(step: int=typer.Argument(1, help='Step to Run', show_default=True, min=1, max=4)):
    """
    Usage:

    Run a Scout Build Step
    $ run <step>
    """
    if step == 1:
        run_step_one()
    elif step == 2:
        run_step_two()
    elif step == 3:
        run_step_three()
    elif step == 4:
        run_step_four()
    else:
        typer.echo(f'Invalid Step: {step}')