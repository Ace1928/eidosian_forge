import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Union
import ray
import ray.util.serialization_addons
from ray.serve._private.common import DeploymentID
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve.schema import ServeApplicationSchema
def deploy_args_to_deployment_info(deployment_name: str, deployment_config_proto_bytes: bytes, replica_config_proto_bytes: bytes, deployer_job_id: Union[str, bytes], route_prefix: Optional[str], docs_path: Optional[str], app_name: Optional[str]=None, ingress: bool=False, **kwargs) -> DeploymentInfo:
    """Takes deployment args passed to the controller after building an application and
    constructs a DeploymentInfo object.
    """
    deployment_config = DeploymentConfig.from_proto_bytes(deployment_config_proto_bytes)
    version = deployment_config.version
    replica_config = ReplicaConfig.from_proto_bytes(replica_config_proto_bytes, deployment_config.needs_pickle())
    if isinstance(deployer_job_id, bytes):
        deployer_job_id = ray.JobID.from_int(int.from_bytes(deployer_job_id, 'little')).hex()
    return DeploymentInfo(actor_name=DeploymentID(deployment_name, app_name).to_replica_actor_class_name(), version=version, deployment_config=deployment_config, replica_config=replica_config, deployer_job_id=deployer_job_id, start_time_ms=int(time.time() * 1000), route_prefix=route_prefix, docs_path=docs_path, ingress=ingress)