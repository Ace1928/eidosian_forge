import json
import logging
import os
from abc import ABC, abstractmethod
import mlflow
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.tracking import MlflowClient
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.utils.file_utils import chdir
def read_run_id(recipe_root):
    run_id_file_path = get_step_output_path(recipe_root, 'train', 'run_id')
    if os.path.exists(run_id_file_path):
        with open(run_id_file_path) as f:
            return f.read().strip()
    return None