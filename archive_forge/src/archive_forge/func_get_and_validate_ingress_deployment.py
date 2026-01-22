import inspect
from collections import OrderedDict
from typing import List
from ray.dag import ClassNode, DAGNode
from ray.dag.function_node import FunctionNode
from ray.dag.utils import _DAGNodeNameGenerator
from ray.experimental.gradio_utils import type_to_string
from ray.serve._private.constants import (
from ray.serve._private.deployment_executor_node import DeploymentExecutorNode
from ray.serve._private.deployment_function_executor_node import (
from ray.serve._private.deployment_function_node import DeploymentFunctionNode
from ray.serve._private.deployment_node import DeploymentNode
from ray.serve.deployment import Deployment, schema_to_deployment
from ray.serve.handle import DeploymentHandle, RayServeHandle
from ray.serve.schema import DeploymentSchema
def get_and_validate_ingress_deployment(deployments: List[Deployment]) -> Deployment:
    """Validation for http route prefixes for a list of deployments in pipeline.

    Ensures:
        1) One and only one ingress deployment with given route prefix.
        2) All other not ingress deployments should have prefix of None.
    """
    ingress_deployments = []
    for deployment in deployments:
        if deployment.route_prefix is not None:
            ingress_deployments.append(deployment)
    if len(ingress_deployments) != 1:
        raise ValueError(f'Only one deployment in an Serve Application or DAG can have non-None route prefix. {len(ingress_deployments)} ingress deployments found: {ingress_deployments}')
    return ingress_deployments[0]