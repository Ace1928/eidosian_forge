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
def deploy_apps(self, config: Union[ServeApplicationSchema, ServeDeploySchema], _blocking: bool=False) -> None:
    """Starts a task on the controller that deploys application(s) from a config.

        Args:
            config: A single-application config (ServeApplicationSchema) or a
                multi-application config (ServeDeploySchema)
            _blocking: Whether to block until the application is running.

        Raises:
            RayTaskError: If the deploy task on the controller fails. This can be
                because a single-app config was deployed after deploying a multi-app
                config, or vice versa.
        """
    ray.get(self._controller.deploy_config.remote(config))
    if _blocking:
        timeout_s = 60
        start = time.time()
        while time.time() - start < timeout_s:
            curr_status = self.get_serve_status()
            if curr_status.app_status.status == ApplicationStatus.RUNNING:
                break
            time.sleep(CLIENT_POLLING_INTERVAL_S)
        else:
            raise TimeoutError(f"Serve application isn't running after {timeout_s}s.")