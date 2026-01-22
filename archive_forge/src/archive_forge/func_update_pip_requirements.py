import logging
import click
from packaging.requirements import InvalidRequirement, Requirement
from mlflow.models import python_api
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.models.model import update_model_requirements
from mlflow.utils import cli_args
from mlflow.utils import env_manager as _EnvManager
@commands.command('update-pip-requirements')
@cli_args.MODEL_URI
@click.argument('operation', type=click.Choice(['add', 'remove']))
@click.argument('requirement_strings', type=str, nargs=-1)
def update_pip_requirements(model_uri, operation, requirement_strings):
    """
    Add or remove requirements from a model's conda.yaml and requirements.txt files.
    If using a remote tracking server, please make sure to set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.

    REQUIREMENT_STRINGS is a list of pip requirements specifiers.
    See below for examples.

    Sample usage:

    .. code::

        # Add requirements using the model's "runs:/" URI

        mlflow models update-pip-requirements -m runs:/<run_id>/<model_path> \\
            add "pandas==1.0.0" "scikit-learn" "mlflow >= 2.8, != 2.9.0"

        # Remove requirements from a local model

        mlflow models update-pip-requirements -m /path/to/local/model \\
            remove "torchvision" "pydantic"

    Note that model registry URIs (i.e. URIs in the form ``models:/``) are not
    supported, as artifacts in the model registry are intended to be read-only.
    Editing requirements is read-only artifact repositories is also not supported.

    If adding requirements, the function will overwrite any existing requirements
    that overlap, or else append the new requirements to the existing list.

    If removing requirements, the function will ignore any version specifiers,
    and remove all the specified package names. Any requirements that are not
    found in the existing files will be ignored.
    """
    try:
        requirements = [Requirement(s.strip().lower()) for s in requirement_strings]
    except InvalidRequirement as e:
        raise click.BadArgumentUsage(f'Invalid requirement: {e}')
    update_model_requirements(model_uri, operation, requirements)
    _logger.info(f'Successfully updated the requirements for the model at {model_uri}!')