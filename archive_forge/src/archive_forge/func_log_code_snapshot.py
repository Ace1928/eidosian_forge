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
def log_code_snapshot(recipe_root: str, run_id: str, artifact_path: str='recipe_snapshot', recipe_config: Optional[Dict[str, Any]]=None) -> None:
    """
    Logs a recipe code snapshot as mlflow artifacts.

    Args:
        recipe_root_path: String file path to the directory where the recipe is defined.
        run_id: Run ID to which the code snapshot is logged.
        artifact_path: Directory within the run's artifact director (default: "snapshots").
        recipe_config: Dict containing the full recipe configuration at runtime.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        recipe_root = pathlib.Path(recipe_root)
        for file_path in (recipe_root.joinpath('recipe.yaml'), recipe_root.joinpath('requirements.txt'), *recipe_root.glob('profiles/*.yaml'), *recipe_root.glob('steps/*.py')):
            if file_path.exists():
                tmp_path = tmpdir.joinpath(file_path.relative_to(recipe_root))
                tmp_path.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(file_path, tmp_path)
        if recipe_config is not None:
            import yaml
            tmp_path = tmpdir.joinpath('runtime/recipe.yaml')
            tmp_path.parent.mkdir(exist_ok=True, parents=True)
            with open(tmp_path, mode='w', encoding='utf-8') as config_file:
                yaml.dump(recipe_config, config_file)
        MlflowClient().log_artifacts(run_id, str(tmpdir), artifact_path=artifact_path)