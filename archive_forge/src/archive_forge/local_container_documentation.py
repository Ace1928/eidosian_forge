import asyncio
import logging
import os
import shlex
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import wandb
from wandb.sdk.launch.environment.abstract import AbstractEnvironment
from wandb.sdk.launch.registry.abstract import AbstractRegistry
from .._project_spec import LaunchProject
from ..builder.build import get_env_vars_dict
from ..errors import LaunchError
from ..utils import (
from .abstract import AbstractRun, AbstractRunner, Status
Return a shell-escaped string from *split_command*.