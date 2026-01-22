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
def run_custom_install_requirements(name: str):
    """
    Helper for running the custom install requirements
    """
    if (custom_installers := config.builds.get(name, {}).get('custom_install')):
        echo(f'{COLOR.BLUE}[{name}]{COLOR.END} Running {COLOR.BOLD}Custom Installers{COLOR.END}')
        for custom in custom_installers:
            if (custom_install := config.custom_installers.get(custom)):
                echo(f'{COLOR.BLUE}[{name}]{COLOR.END} Running {COLOR.BOLD}{custom}{COLOR.END}')
                for cmdstr in custom_install:
                    os.system(cmdstr)