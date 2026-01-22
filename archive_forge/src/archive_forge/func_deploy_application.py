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
def deploy_application(self, name, deployments: List[Dict], _blocking: bool=True):
    deployment_args_list = []
    for deployment in deployments:
        deployment_args = get_deploy_args(deployment['name'], replica_config=deployment['replica_config'], ingress=deployment['ingress'], deployment_config=deployment['deployment_config'], version=deployment['version'], route_prefix=deployment['route_prefix'], docs_path=deployment['docs_path'])
        deployment_args_proto = DeploymentArgs()
        deployment_args_proto.deployment_name = deployment_args['deployment_name']
        deployment_args_proto.deployment_config = deployment_args['deployment_config_proto_bytes']
        deployment_args_proto.replica_config = deployment_args['replica_config_proto_bytes']
        deployment_args_proto.deployer_job_id = deployment_args['deployer_job_id']
        if deployment_args['route_prefix']:
            deployment_args_proto.route_prefix = deployment_args['route_prefix']
        deployment_args_proto.ingress = deployment_args['ingress']
        if deployment_args['docs_path']:
            deployment_args_proto.docs_path = deployment_args['docs_path']
        deployment_args_list.append(deployment_args_proto.SerializeToString())
    ray.get(self._controller.deploy_application.remote(name, deployment_args_list))
    if _blocking:
        self._wait_for_application_running(name)
        for deployment in deployments:
            deployment_name = deployment['name']
            tag = f'component=serve deployment={deployment_name}'
            url = deployment['url']
            version = deployment['version']
            self.log_deployment_ready(deployment_name, version, url, tag)