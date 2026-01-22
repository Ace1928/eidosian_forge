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
def get_local_python_deps(dir: str, filename: str='requirements.local.txt') -> Optional[str]:
    try:
        env = os.environ
        with open(os.path.join(dir, filename), 'w') as f:
            subprocess.call(['pip', 'freeze'], env=env, stdout=f)
        return filename
    except subprocess.CalledProcessError as e:
        wandb.termerror(f'Command failed: {e}')
        return None