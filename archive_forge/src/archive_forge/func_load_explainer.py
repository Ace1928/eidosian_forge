import os
import tempfile
import types
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional
import numpy as np
import yaml
import mlflow
import mlflow.utils.autologging_utils
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_package_name
from mlflow.utils.uri import append_to_uri_path
def load_explainer(model_uri):
    """
    Load a SHAP explainer from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.

    Returns:
        A SHAP explainer.
    """
    explainer_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=explainer_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(explainer_path, flavor_conf)
    explainer_artifacts_path = os.path.join(explainer_path, flavor_conf['serialized_explainer'])
    underlying_model_flavor = flavor_conf['underlying_model_flavor']
    model = None
    if underlying_model_flavor != _UNKNOWN_MODEL_FLAVOR:
        underlying_model_path = os.path.join(explainer_path, _UNDERLYING_MODEL_SUBPATH)
        if underlying_model_flavor == mlflow.sklearn.FLAVOR_NAME:
            model = mlflow.sklearn._load_pyfunc(underlying_model_path).predict
        elif underlying_model_flavor == mlflow.pytorch.FLAVOR_NAME:
            model = mlflow.pytorch._load_model(os.path.join(underlying_model_path, 'data'))
    return _load_explainer(explainer_file=explainer_artifacts_path, model=model)