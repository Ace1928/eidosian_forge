import json
import sys
from inspect import signature
import click
from mlflow.deployments import interface
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_CONFIG
from mlflow.utils import cli_args
from mlflow.utils.annotations import experimental
from mlflow.utils.proto_json_utils import NumpyEncoder, _get_jsonable_obj
@commands.command('list')
@optional_endpoint_param
@target_details
def list_deployment(target, endpoint):
    """
    List the names of all model deployments in the specified target. These names can be used with
    the `delete`, `update`, and `get` commands.
    """
    client = interface.get_deploy_client(target)
    sig = signature(client.list_deployments)
    if 'endpoint' in sig.parameters:
        ids = client.list_deployments(endpoint=endpoint)
    else:
        ids = client.list_deployments()
    click.echo(f'List of all deployments:\n{ids}')