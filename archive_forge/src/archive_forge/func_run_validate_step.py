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
def run_validate_step(name: str):
    """
    Helper for running the validation step
    """
    kind = config.builds[name]['kind']
    echo(f'{COLOR.BLUE}[{kind}]{COLOR.END} Validating {COLOR.BOLD}{name}{COLOR.END}')
    if config.builds[name].get('validate'):
        validate_cmd = config.builds[name]['validate']
        from lazyops.utils.lazy import lazy_import
        validate_func = lazy_import(validate_cmd)
        validate_func()
    elif name == 'server':
        with contextlib.suppress(Exception):
            from lazyops.utils.lazy import lazy_import
            validate_func = lazy_import(f'{config.app_name}.app.main.validate_build')
            validate_func()
    echo(f'{COLOR.BLUE}[{kind}]{COLOR.END} Completed {COLOR.BOLD}{config.app_name}{COLOR.END} Validation')