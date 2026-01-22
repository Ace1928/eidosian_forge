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
def strip_resource_args_and_template_vars(launch_spec: Dict[str, Any]) -> None:
    wandb.termwarn('Launch spec contains both resource_args and template_variables, only one can be set. Using template_variables.')
    if launch_spec.get('resource_args', None) and launch_spec.get('template_variables', None):
        launch_spec['resource_args'] = None