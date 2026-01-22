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
def init_build_service(name: str):
    """
    Helper for initializing the build service
    """
    enabled = config.has_service(name)
    kind = config.builds[name]['kind']
    if enabled:
        echo(f'{COLOR.BLUE}[{kind}]{COLOR.END} Initializing {COLOR.BOLD}{name}{COLOR.END}')
        if config.builds[name].get('init'):
            init_cmd = config.builds[name]['init']
            from lazyops.utils.lazy import lazy_import
            init_func = lazy_import(init_cmd)
            init_func()
        elif name == 'server':
            with contextlib.suppress(Exception):
                from lazyops.utils.lazy import lazy_import
                init_func = lazy_import(f'{config.app_name}.app.main.init_build')
                init_func()
        echo(f'{COLOR.BLUE}[{kind}]{COLOR.END} Completed {COLOR.BOLD}{name}{COLOR.END} Initialization')
        add_to_env(f'{name.upper()}_{kind.upper()}_ENABLED', True)
    else:
        add_to_env(f'{name.upper()}_{kind.upper()}_ENABLED', False)