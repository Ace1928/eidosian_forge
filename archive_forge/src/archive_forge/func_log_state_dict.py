import atexit
import importlib
import logging
import os
import posixpath
import shutil
import warnings
from functools import partial
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_DEFAULT_PREDICTION_DEVICE
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.pytorch import pickle_module as mlflow_pytorch_pickle_module
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.checkpoint_utils import download_checkpoint_artifact
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def log_state_dict(state_dict, artifact_path, **kwargs):
    """
    Log a state_dict as an MLflow artifact for the current run.

    .. warning::
        This function just logs a state_dict as an artifact and doesn't generate
        an :ref:`MLflow Model <models>`.

    Args:
        state_dict: state_dict to be saved.
        artifact_path: Run-relative artifact path.
        kwargs: kwargs to pass to ``torch.save``.

    .. code-block:: python
        :caption: Example

        # Log a model as a state_dict
        with mlflow.start_run():
            state_dict = model.state_dict()
            mlflow.pytorch.log_state_dict(state_dict, artifact_path="model")

        # Log a checkpoint as a state_dict
        with mlflow.start_run():
            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
            }
            mlflow.pytorch.log_state_dict(state_dict, artifact_path="checkpoint")
    """
    with TempDir() as tmp:
        local_path = tmp.path()
        save_state_dict(state_dict=state_dict, path=local_path, **kwargs)
        mlflow.log_artifacts(local_path, artifact_path)