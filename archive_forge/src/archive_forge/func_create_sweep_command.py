import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import yaml
import wandb
from wandb import util
from wandb.sdk.launch.errors import LaunchError
def create_sweep_command(command: Optional[List]=None) -> List:
    """Return sweep command, filling in environment variable macros."""
    command = command or DEFAULT_SWEEP_COMMAND
    for i, chunk in enumerate(command):
        if SWEEP_COMMAND_ENV_VAR_REGEX.search(str(chunk)):
            matches = list(SWEEP_COMMAND_ENV_VAR_REGEX.finditer(chunk))
            for m in matches[::-1]:
                _var: str = os.environ.get(m.group(1), m.group(1))
                command[i] = f'{command[i][:m.start()]}{_var}{command[i][m.end():]}'
    return command