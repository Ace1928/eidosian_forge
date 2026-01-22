import enum
import logging
import os
import tempfile
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
import wandb
import wandb.docker as docker
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch import utils
from wandb.sdk.lib.runid import generate_id
from .errors import LaunchError
from .utils import LOG_PREFIX, recursive_macro_sub
def update_entrypoint_path(self, new_path: str) -> None:
    """Updates the entrypoint path to a new path."""
    if len(self.command) == 2 and self.command[0] in ['python', 'bash']:
        self.command[1] = new_path