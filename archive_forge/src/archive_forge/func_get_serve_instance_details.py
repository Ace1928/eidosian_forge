import asyncio
import logging
import marshal
import os
import pickle
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import ray
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.utils import run_background_task
from ray._raylet import GcsClient
from ray.actor import ActorHandle
from ray.serve._private.application_state import ApplicationStateManager
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve._private.default_impl import create_cluster_node_info_cache
from ray.serve._private.deploy_utils import deploy_args_to_deployment_info
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_state import DeploymentStateManager
from ray.serve._private.endpoint_state import EndpointState
from ray.serve._private.logging_utils import (
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.proxy_state import ProxyStateManager
from ray.serve._private.storage.kv_store import RayInternalKVStore
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.config import HTTPOptions, gRPCOptions
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import EndpointInfo as EndpointInfoProto
from ray.serve.generated.serve_pb2 import EndpointSet
from ray.serve.schema import (
from ray.util import metrics
def get_serve_instance_details(self) -> Dict:
    """Gets details on all applications on the cluster and system-level info.

        The information includes application and deployment statuses, config options,
        error messages, etc.

        Returns:
            Dict that follows the format of the schema ServeInstanceDetails. Currently,
            there is a value set for every field at all schema levels, except for the
            route_prefix in the deployment_config for each deployment.
        """
    http_config = self.get_http_config()
    grpc_config = self.get_grpc_config()
    applications = {}
    for app_name, app_status_info in self.application_state_manager.list_app_statuses().items():
        applications[app_name] = ApplicationDetails(name=app_name, route_prefix=self.application_state_manager.get_route_prefix(app_name), docs_path=self.get_docs_path(app_name), status=app_status_info.status, message=app_status_info.message, last_deployed_time_s=app_status_info.deployment_timestamp, deployed_app_config=self.get_app_config(app_name), deployments=self.application_state_manager.list_deployment_details(app_name))
    http_options = HTTPOptionsSchema.parse_obj(http_config.dict(exclude_unset=True))
    grpc_options = gRPCOptionsSchema.parse_obj(grpc_config.dict(exclude_unset=True))
    return ServeInstanceDetails(target_capacity=self._target_capacity, controller_info=self._actor_details, proxy_location=http_config.location, http_options=http_options, grpc_options=grpc_options, proxies=self.proxy_state_manager.get_proxy_details() if self.proxy_state_manager else None, applications=applications).dict(exclude_unset=True)