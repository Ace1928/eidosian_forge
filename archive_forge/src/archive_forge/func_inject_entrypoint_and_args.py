import asyncio
import base64
import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import yaml
import wandb
from wandb.apis.internal import Api
from wandb.sdk.launch.agent.agent import LaunchAgent
from wandb.sdk.launch.environment.abstract import AbstractEnvironment
from wandb.sdk.launch.registry.abstract import AbstractRegistry
from wandb.sdk.launch.registry.azure_container_registry import AzureContainerRegistry
from wandb.sdk.launch.registry.local_registry import LocalRegistry
from wandb.sdk.launch.runner.abstract import Status
from wandb.sdk.launch.runner.kubernetes_monitor import (
from wandb.util import get_module
from .._project_spec import EntryPoint, LaunchProject
from ..builder.build import get_env_vars_dict
from ..errors import LaunchError
from ..utils import (
from .abstract import AbstractRun, AbstractRunner
import kubernetes_asyncio  # type: ignore # noqa: E402
from kubernetes_asyncio import client  # noqa: E402
from kubernetes_asyncio.client.api.batch_v1_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.api.core_v1_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.api.custom_objects_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.models.v1_secret import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.rest import ApiException  # type: ignore # noqa: E402
def inject_entrypoint_and_args(containers: List[dict], entry_point: Optional[EntryPoint], override_args: List[str], should_override_entrypoint: bool) -> None:
    """Inject the entrypoint and args into the containers.

    Arguments:
        containers: The containers to inject the entrypoint and args into.
        entry_point: The entrypoint to inject.
        override_args: The args to inject.
        should_override_entrypoint: Whether to override the entrypoint.

    Returns:
        None
    """
    for i in range(len(containers)):
        if override_args:
            containers[i]['args'] = override_args
        if entry_point and (not containers[i].get('command') or should_override_entrypoint):
            containers[i]['command'] = entry_point.command