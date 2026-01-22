import asyncio
import pprint
from typing import Any, Dict, List, Optional, Union
import wandb
import wandb.apis.public as public
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch.builder.build import build_image_from_project
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.utils import (
from ._project_spec import LaunchProject
def push_to_queue(api: Api, queue_name: str, launch_spec: Dict[str, Any], template_variables: Optional[dict], project_queue: str, priority: Optional[int]=None) -> Any:
    return api.push_to_run_queue(queue_name, launch_spec, template_variables, project_queue, priority)