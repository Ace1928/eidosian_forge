import inspect
import logging
from types import FunctionType
from typing import Any, Dict, Tuple, Union
import ray
from ray._private.pydantic_compat import is_subclass_of_base_model
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.usage import usage_lib
from ray.actor import ActorHandle
from ray.serve._private.client import ServeControllerClient
from ray.serve._private.constants import (
from ray.serve._private.controller import ServeController
from ray.serve.config import HTTPOptions, gRPCOptions
from ray.serve.context import _get_global_client, _set_global_client
from ray.serve.deployment import Application, Deployment
from ray.serve.exceptions import RayServeException
from ray.serve.schema import LoggingConfig
def get_deployment(name: str, app_name: str=''):
    """Dynamically fetch a handle to a Deployment object.

    Args:
        name: name of the deployment. This must have already been
        deployed.

    Returns:
        Deployment
    """
    try:
        deployment_info, route_prefix = _get_global_client().get_deployment_info(name, app_name)
    except KeyError:
        if len(app_name) == 0:
            msg = f'Deployment {name} was not found. Did you call Deployment.deploy()? Note that `serve.get_deployment()` can only be used to fetch a deployment that was deployed using the 1.x API `Deployment.deploy()`. If you want to fetch a handle to an application deployed through `serve.run` or through a Serve config, please use `serve.get_app_handle()` instead.'
        else:
            msg = f'Deployment {name} in application {app_name} was not found.'
        raise KeyError(msg)
    return Deployment(name, deployment_info.deployment_config, deployment_info.replica_config, version=deployment_info.version, route_prefix=route_prefix, _internal=True)