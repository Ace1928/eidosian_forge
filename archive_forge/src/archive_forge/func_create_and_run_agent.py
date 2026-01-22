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
def create_and_run_agent(api: Api, config: Dict[str, Any]) -> None:
    try:
        from wandb.sdk.launch.agent import config as agent_config
    except ModuleNotFoundError:
        raise LaunchError('wandb launch-agent requires pydantic to be installed. Please install with `pip install wandb[launch]`')
    try:
        agent_config.AgentConfig(**config)
    except agent_config.ValidationError as e:
        errors = e.errors()
        for error in errors:
            loc = '.'.join([str(x) for x in error.get('loc', [])])
            msg = f'Agent config error in field {loc}'
            value = error.get('input')
            if not isinstance(value, dict):
                msg += f' (value: {value})'
            msg += f': {error['msg']}'
            wandb.termerror(msg)
        raise LaunchError('Invalid launch agent config')
    agent = LaunchAgent(api, config)
    try:
        asyncio.run(agent.loop())
    except asyncio.CancelledError:
        pass