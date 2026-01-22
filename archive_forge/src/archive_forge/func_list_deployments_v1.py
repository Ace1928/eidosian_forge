import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import ray
from ray.actor import ActorHandle
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import (
from ray.serve._private.controller import ServeController
from ray.serve._private.deploy_utils import get_deploy_args
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve.config import HTTPOptions
from ray.serve.exceptions import RayServeException
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import StatusOverview as StatusOverviewProto
from ray.serve.handle import DeploymentHandle, RayServeHandle, RayServeSyncHandle
from ray.serve.schema import LoggingConfig, ServeApplicationSchema, ServeDeploySchema
@_ensure_connected
def list_deployments_v1(self) -> Dict[str, Tuple[DeploymentInfo, str]]:
    """Gets the current information about all 1.x deployments."""
    deployment_route_list = DeploymentRouteList.FromString(ray.get(self._controller.list_deployments_v1.remote()))
    return {deployment_route.deployment_info.name: (DeploymentInfo.from_proto(deployment_route.deployment_info), deployment_route.route if deployment_route.route != '' else None) for deployment_route in deployment_route_list.deployment_routes}