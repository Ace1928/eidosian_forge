import logging
import os
import pathlib
import posixpath
from typing import Any, Dict, Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import read_yaml, render_and_merge_yaml
def get_recipe_config(recipe_root_path: Optional[str]=None, profile: Optional[str]=None) -> Dict[str, Any]:
    """
    Obtains a dictionary representation of the configuration for the specified recipe.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem. If unspecified, the recipe root directory is resolved from the current
            working directory.
        profile: The name of the profile under the `profiles` directory to use, e.g. "dev" to
            use configs from "profiles/dev.yaml".

    Raises:
        MlflowException: If the specified ``recipe_root_path`` is not a recipe root directory
            or if ``recipe_root_path`` is ``None`` and the current working directory does not
            correspond to a recipe.

    Returns:
        The configuration of the specified recipe.
    """
    recipe_root_path = recipe_root_path or get_recipe_root_path()
    _verify_is_recipe_root_directory(recipe_root_path=recipe_root_path)
    try:
        if profile:
            profile_relpath = posixpath.join(_RECIPE_PROFILE_DIR, f'{profile}.yaml')
            profile_file_path = os.path.join(recipe_root_path, _RECIPE_PROFILE_DIR, f'{profile}.yaml')
            if not os.path.exists(profile_file_path):
                raise MlflowException(f"Did not find the YAML configuration file for the specified profile '{profile}' at expected path '{profile_file_path}'.", error_code=INVALID_PARAMETER_VALUE)
            return render_and_merge_yaml(recipe_root_path, _RECIPE_CONFIG_FILE_NAME, profile_relpath)
        else:
            return read_yaml(recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    except MlflowException:
        raise
    except Exception as e:
        raise MlflowException('Failed to read recipe configuration. Please verify that the `recipe.yaml` configuration file and the YAML configuration file for the selected profile are syntactically correct and that the specified profile provides all required values for template substitutions defined in `recipe.yaml`.', error_code=INVALID_PARAMETER_VALUE) from e