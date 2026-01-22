import asyncio
import logging
from typing import Any, Dict, List, Optional, cast
import wandb
from wandb.apis.internal import Api
from wandb.sdk.launch.environment.aws_environment import AwsEnvironment
from wandb.sdk.launch.errors import LaunchError
from .._project_spec import EntryPoint, LaunchProject, get_entry_point_command
from ..builder.build import get_env_vars_dict
from ..registry.abstract import AbstractRegistry
from ..utils import (
from .abstract import AbstractRun, AbstractRunner, Status
def get_role_arn(sagemaker_args: Dict[str, Any], backend_config: Dict[str, Any], account_id: str) -> str:
    """Get the role arn from the sagemaker args or the backend config."""
    role_arn = sagemaker_args.get('RoleArn') or sagemaker_args.get('role_arn')
    if role_arn is None:
        role_arn = backend_config.get('runner', {}).get('role_arn')
    if role_arn is None or not isinstance(role_arn, str):
        raise LaunchError('AWS sagemaker require a string RoleArn set this by adding a `RoleArn` key to the sagemakerfield of resource_args')
    if role_arn.startswith('arn:aws:iam::'):
        return role_arn
    return f'arn:aws:iam::{account_id}:role/{role_arn}'