import asyncio
import json
import logging
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast
import click
import wandb
import wandb.docker as docker
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.wandb_reference import WandbReference
from wandb.sdk.wandb_config import Config
from .builder.templates._wandb_bootstrap import (
def validate_wandb_python_deps(requirements_file: Optional[str], dir: str) -> None:
    """Warn if local python dependencies differ from wandb requirements.txt."""
    if requirements_file is not None:
        requirements_path = os.path.join(dir, requirements_file)
        with open(requirements_path) as f:
            wandb_python_deps: List[str] = f.read().splitlines()
        local_python_file = get_local_python_deps(dir)
        if local_python_file is not None:
            local_python_deps_path = os.path.join(dir, local_python_file)
            with open(local_python_deps_path) as f:
                local_python_deps: List[str] = f.read().splitlines()
            diff_pip_requirements(wandb_python_deps, local_python_deps)
            return
    _logger.warning('Unable to validate local python dependencies')