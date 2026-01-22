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
def transform_ray_dag_to_serve_dag(dag_node: DAGNode, node_name_generator: _DAGNodeNameGenerator, app_name: str):
    """
    Transform a Ray DAG to a Serve DAG. Map ClassNode to DeploymentNode with
    ray decorated body passed in.
    """
    if isinstance(dag_node, ClassNode):
        deployment_name = node_name_generator.get_node_name(dag_node)

        def replace_with_handle(node):
            if isinstance(node, DeploymentNode) or isinstance(node, DeploymentFunctionNode):
                if RAY_SERVE_ENABLE_NEW_HANDLE_API:
                    return DeploymentHandle(node._deployment.name, app_name, sync=False)
                else:
                    return RayServeHandle(node._deployment.name, app_name, sync=False)
            elif isinstance(node, DeploymentExecutorNode):
                return node._deployment_handle
        replaced_deployment_init_args, replaced_deployment_init_kwargs = dag_node.apply_functional([dag_node.get_args(), dag_node.get_kwargs()], predictate_fn=lambda node: isinstance(node, (DeploymentNode, DeploymentFunctionNode, DeploymentExecutorNode, DeploymentFunctionExecutorNode)), apply_fn=replace_with_handle)
        deployment_schema: DeploymentSchema = dag_node._bound_other_args_to_resolve['deployment_schema']
        deployment_shell: Deployment = schema_to_deployment(deployment_schema)
        if inspect.isclass(dag_node._body) and deployment_shell.name != dag_node._body.__name__:
            deployment_name = deployment_shell.name
        if deployment_shell.route_prefix is None or deployment_shell.route_prefix != f'/{deployment_shell.name}':
            route_prefix = deployment_shell.route_prefix
        else:
            route_prefix = f'/{deployment_name}'
        deployment = deployment_shell.options(func_or_class=dag_node._body, name=deployment_name, route_prefix=route_prefix, _init_args=replaced_deployment_init_args, _init_kwargs=replaced_deployment_init_kwargs, _internal=True)
        return DeploymentNode(deployment, app_name, dag_node.get_args(), dag_node.get_kwargs(), dag_node.get_options(), other_args_to_resolve=dag_node.get_other_args_to_resolve())
    elif isinstance(dag_node, FunctionNode) and dag_node.get_other_args_to_resolve().get('is_from_serve_deployment'):
        deployment_name = node_name_generator.get_node_name(dag_node)
        other_args_to_resolve = dag_node.get_other_args_to_resolve()
        if 'return' in dag_node._body.__annotations__:
            other_args_to_resolve['result_type_string'] = type_to_string(dag_node._body.__annotations__['return'])
        if 'deployment_schema' in dag_node._bound_other_args_to_resolve:
            schema = dag_node._bound_other_args_to_resolve['deployment_schema']
            if inspect.isfunction(dag_node._body) and schema.name != dag_node._body.__name__:
                deployment_name = schema.name
        return DeploymentFunctionNode(dag_node._body, deployment_name, app_name, dag_node.get_args(), dag_node.get_kwargs(), dag_node.get_options(), other_args_to_resolve=other_args_to_resolve)
    else:
        return dag_node