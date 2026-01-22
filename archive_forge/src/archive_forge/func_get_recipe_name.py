import logging
import os
import pathlib
import posixpath
from typing import Any, Dict, Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import read_yaml, render_and_merge_yaml
def get_recipe_name(recipe_root_path: Optional[str]=None) -> str:
    """
    Obtains the name of the specified recipe or of the recipe corresponding to the current
    working directory.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem. If unspecified, the recipe root directory is resolved from the current
            working directory.

    Raises:
        MlflowException: If the specified ``recipe_root_path`` is not a recipe root
            directory or if ``recipe_root_path`` is ``None`` and the current working directory
            does not correspond to a recipe.

    Returns:
        The name of the specified recipe.
    """
    recipe_root_path = recipe_root_path or get_recipe_root_path()
    _verify_is_recipe_root_directory(recipe_root_path=recipe_root_path)
    return os.path.basename(recipe_root_path)