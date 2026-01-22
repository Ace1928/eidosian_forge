import inspect
import itertools
import logging
import os
from typing import Any, Dict, Optional
import yaml
import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.validation import _is_numeric
def wrapper_fit(original, self, *args, **kwargs):
    should_autolog = False
    if AutologHelpers.should_autolog:
        AutologHelpers.should_autolog = False
        should_autolog = True
    try:
        if should_autolog:
            log_fn_args_as_params(original, args, kwargs)
        model = original(self, *args, **kwargs)
        if should_autolog:
            if get_autologging_config(FLAVOR_NAME, 'log_models', True):
                global _save_model_called_from_autolog
                _save_model_called_from_autolog = True
                registered_model_name = get_autologging_config(FLAVOR_NAME, 'registered_model_name', None)
                try:
                    log_model(model, artifact_path='model', registered_model_name=registered_model_name)
                finally:
                    _save_model_called_from_autolog = False
            if isinstance(model, statsmodels.base.wrapper.ResultsWrapper):
                metrics_dict = _get_autolog_metrics(model)
                mlflow.log_metrics(metrics_dict)
                model_summary = model.summary().as_text()
                mlflow.log_text(model_summary, 'model_summary.txt')
        return model
    finally:
        if should_autolog:
            AutologHelpers.should_autolog = True