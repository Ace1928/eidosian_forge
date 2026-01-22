import logging
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray import cloudpickle
from ray._private.utils import import_attr
from ray.exceptions import RuntimeEnvSetupError
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.deploy_utils import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_state import DeploymentStateManager
from ray.serve._private.endpoint_state import EndpointState
from ray.serve._private.storage.kv_store import KVStoreBase
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.exceptions import RayServeException
from ray.serve.generated.serve_pb2 import DeploymentLanguage
from ray.serve.schema import DeploymentDetails, ServeApplicationSchema
from ray.types import ObjectRef
@ray.remote(num_cpus=0, max_calls=1)
def build_serve_application(import_path: str, config_deployments: List[str], code_version: str, name: str, args: Dict) -> Tuple[List[Dict], Optional[str]]:
    """Import and build a Serve application.

    Args:
        import_path: import path to top-level bound deployment.
        config_deployments: list of deployment names specified in config
            with deployment override options. This is used to check that
            all deployments specified in the config are valid.
        code_version: code version inferred from app config. All
            deployment versions are set to this code version.
        name: application name. If specified, application will be deployed
            without removing existing applications.
        args: Arguments to be passed to the application builder.
        logging_config: The application logging config, if deployment logging
            config is not set, application logging config will be applied to the
            deployment logging config.
    Returns:
        Deploy arguments: a list of deployment arguments if application
            was built successfully, otherwise None.
        Error message: a string if an error was raised, otherwise None.
    """
    try:
        from ray.serve._private.api import call_app_builder_with_args_if_necessary
        from ray.serve._private.deployment_graph_build import build as pipeline_build
        from ray.serve._private.deployment_graph_build import get_and_validate_ingress_deployment
        app = call_app_builder_with_args_if_necessary(import_attr(import_path), args)
        deployments = pipeline_build(app._get_internal_dag_node(), name)
        ingress = get_and_validate_ingress_deployment(deployments)
        deploy_args_list = []
        for deployment in deployments:
            is_ingress = deployment.name == ingress.name
            deploy_args_list.append(get_deploy_args(name=deployment._name, replica_config=deployment._replica_config, ingress=is_ingress, deployment_config=deployment._deployment_config, version=code_version, route_prefix=deployment.route_prefix, docs_path=deployment._docs_path))
        return (deploy_args_list, None)
    except KeyboardInterrupt:
        logger.info('Existing config deployment request terminated.')
        return (None, None)
    except Exception:
        return (None, traceback.format_exc())