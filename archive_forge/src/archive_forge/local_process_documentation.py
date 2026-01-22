import logging
import shlex
from typing import Any, List, Optional
import wandb
from .._project_spec import LaunchProject, get_entry_point_command
from ..builder.build import get_env_vars_dict
from ..errors import LaunchError
from ..utils import (
from .abstract import AbstractRun, AbstractRunner
from .local_container import _run_entry_point
Runner class, uses a project to create a LocallySubmittedRun.

    LocalProcessRunner is very similar to a LocalContainerRunner, except it does not
    run the command inside a docker container. Instead, it runs the
    command specified as a process directly on the bare metal machine.

    