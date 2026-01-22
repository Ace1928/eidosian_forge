import hashlib
import logging
import os
import pathlib
import re
import shutil
from typing import Dict, List
from mlflow.environment_variables import (
from mlflow.recipes.step import BaseStep, StepStatus
from mlflow.utils.file_utils import read_yaml, write_yaml
from mlflow.utils.process import _exec_cmd
def get_step_output_path(recipe_root_path: str, step_name: str, relative_path: str) -> str:
    """
    Obtains the absolute path of the specified step output on the local filesystem. Does
    not check the existence of the output.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.
        step_name: The name of the recipe step containing the specified output.
        relative_path: The relative path of the output within the output directory
            of the specified recipe step.

    Returns:
        The absolute path of the step output on the local filesystem, which may or may
        not exist.
    """
    execution_dir_path = get_or_create_base_execution_directory(recipe_root_path=recipe_root_path)
    step_outputs_path = _get_step_output_directory_path(execution_directory_path=execution_dir_path, step_name=step_name)
    return os.path.abspath(os.path.join(step_outputs_path, relative_path))