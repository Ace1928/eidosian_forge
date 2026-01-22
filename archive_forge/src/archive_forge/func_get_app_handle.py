import collections
import inspect
import logging
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
from fastapi import APIRouter, FastAPI
import ray
from ray import cloudpickle
from ray._private.serialization import pickle_dumps
from ray.dag import DAGNode
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import (
from ray.serve._private.deployment_graph_build import build as pipeline_build
from ray.serve._private.deployment_graph_build import (
from ray.serve._private.http_util import (
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.config import (
from ray.serve.context import (
from ray.serve.deployment import Application, Deployment
from ray.serve.exceptions import RayServeException
from ray.serve.handle import DeploymentHandle
from ray.serve.multiplex import _ModelMultiplexWrapper
from ray.serve.schema import LoggingConfig, ServeInstanceDetails, ServeStatus
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.serve._private import api as _private_api  # isort:skip
@PublicAPI(stability='alpha')
def get_app_handle(name: str) -> DeploymentHandle:
    """Get a handle to the application's ingress deployment by name.

    Args:
        name: Name of application to get a handle to.

    Raises:
        RayServeException: If no Serve controller is running, or if the
            application does not exist.

    .. code-block:: python

            import ray
            from ray import serve

            @serve.deployment
            def f(val: int) -> int:
                return val * 2

            serve.run(f.bind(), name="my_app")
            handle = serve.get_app_handle("my_app")
            assert handle.remote(3).result() == 6
    """
    client = _get_global_client()
    ingress = ray.get(client._controller.get_ingress_deployment_name.remote(name))
    if ingress is None:
        raise RayServeException(f"Application '{name}' does not exist.")
    ServeUsageTag.SERVE_GET_APP_HANDLE_API_USED.record('1')
    sync = _get_internal_replica_context() is None
    return client.get_handle(ingress, name, sync=sync, use_new_handle_api=True)