import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import yaml
import wandb
from wandb.apis.internal import Api
from . import loader
from ._project_spec import LaunchProject
from .agent import LaunchAgent
from .builder.build import construct_agent_configs
from .environment.local_environment import LocalEnvironment
from .errors import ExecutionError, LaunchError
from .runner.abstract import AbstractRun
from .utils import (
def set_launch_logfile(logfile: str) -> None:
    """Set the logfile for the launch agent."""
    _launch_logger = logging.getLogger('wandb.sdk.launch')
    if logfile == '-':
        logfile_stream = sys.stdout
    else:
        try:
            logfile_stream = open(logfile, 'w')
        except Exception as e:
            wandb.termerror(f'Could not open {logfile} for writing logs. Please check the path and permissions.\nError: {e}')
            return
    wandb.termlog(f'Internal agent logs printing to {('stdout' if logfile == '-' else logfile)}. ')
    handler = logging.StreamHandler(logfile_stream)
    handler.formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(threadName)-10s:%(process)d [%(filename)s:%(funcName)s():%(lineno)s] %(message)s')
    _launch_logger.addHandler(handler)
    _launch_logger.log(logging.INFO, 'Internal agent logs printing to %s', logfile)