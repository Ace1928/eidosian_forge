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
def get_execution_state(step):
    return step.get_execution_state(output_directory=_get_step_output_directory_path(execution_directory_path=execution_dir_path, step_name=step.name))