import json
import logging
import pathlib
import shutil
import tempfile
import uuid
from typing import Any, Dict, Optional
import mlflow
from mlflow.environment_variables import MLFLOW_RUN_CONTEXT
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.recipes.utils import get_recipe_name
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.tracking.fluent import set_experiment as fluent_set_experiment
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import path_to_local_file_uri, path_to_local_sqlite_uri
from mlflow.utils.git_utils import get_git_branch, get_git_commit, get_git_repo_url
from mlflow.utils.mlflow_tags import (
def get_recipe_tracking_config(recipe_root_path: str, recipe_config: Dict[str, Any]) -> TrackingConfig:
    """
    Obtains the MLflow Tracking configuration for the specified recipe.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.
        recipe_config: The configuration of the specified recipe.

    Returns:
        A ``TrackingConfig`` instance containing MLflow Tracking information for the
        specified recipe, including Tracking URI, Experiment name, and more.
    """
    if is_in_databricks_runtime():
        default_tracking_uri = 'databricks'
        default_artifact_location = None
    else:
        mlflow_metadata_base_path = pathlib.Path(recipe_root_path) / 'metadata' / 'mlflow'
        mlflow_metadata_base_path.mkdir(exist_ok=True, parents=True)
        default_tracking_uri = path_to_local_sqlite_uri(path=str((mlflow_metadata_base_path / 'mlruns.db').resolve()))
        default_artifact_location = path_to_local_file_uri(path=str((mlflow_metadata_base_path / 'mlartifacts').resolve()))
    tracking_config = recipe_config.get('experiment', {})
    config_obj_kwargs = {'run_name': _get_run_name(tracking_config.get('run_name_prefix')), 'tracking_uri': tracking_config.get('tracking_uri', default_tracking_uri), 'artifact_location': tracking_config.get('artifact_location', default_artifact_location)}
    experiment_name = tracking_config.get('name')
    if experiment_name is not None:
        return TrackingConfig(experiment_name=experiment_name, **config_obj_kwargs)
    experiment_id = tracking_config.get('id')
    if experiment_id is not None:
        return TrackingConfig(experiment_id=experiment_id, **config_obj_kwargs)
    experiment_id = _get_experiment_id()
    if experiment_id != DEFAULT_EXPERIMENT_ID:
        return TrackingConfig(experiment_id=experiment_id, **config_obj_kwargs)
    return TrackingConfig(experiment_name=get_recipe_name(recipe_root_path=recipe_root_path), **config_obj_kwargs)