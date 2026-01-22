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
def set_entry_point(self, command: List[str]) -> 'EntryPoint':
    """Add an entry point to the project."""
    assert self._entry_point is None, 'Cannot set entry point twice. Use LaunchProject.override_entrypoint'
    new_entrypoint = EntryPoint(name=command[-1], command=command)
    self._entry_point = new_entrypoint
    return new_entrypoint