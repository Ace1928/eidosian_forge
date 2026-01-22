import functools
import inspect
import logging
import os
import pickle
import weakref
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Dict, Optional
import numpy as np
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _inspect_original_var_name, gorilla
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.mlflow_tags import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def patched_predict(original, self, *args, **kwargs):
    """
        In `patched_predict`, register the prediction result instance with the run id and
         eval dataset name. e.g.
        ```
        prediction_result = model_1.predict(eval_X)
        ```
        then we need register the following relationship into the `_AUTOLOGGING_METRICS_MANAGER`:
        id(prediction_result) --> (eval_dataset_name, run_id)

        Note: we cannot set additional attributes "eval_dataset_name" and "run_id" into
        the prediction_result object, because certain dataset type like numpy does not support
        additional attribute assignment.
        """
    run_id = _AUTOLOGGING_METRICS_MANAGER.get_run_id_for_model(self)
    if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics() and run_id:
        with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
            predict_result = original(self, *args, **kwargs)
        eval_dataset = get_instance_method_first_arg_value(original, args, kwargs)
        eval_dataset_name = _AUTOLOGGING_METRICS_MANAGER.register_prediction_input_dataset(self, eval_dataset)
        _AUTOLOGGING_METRICS_MANAGER.register_prediction_result(run_id, eval_dataset_name, predict_result)
        if log_datasets:
            try:
                context_tags = context_registry.resolve_tags()
                source = CodeDatasetSource(context_tags)
                dataset = _create_dataset(eval_dataset, source)
                if dataset:
                    tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value='eval')]
                    dataset_input = DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags)
                    client = mlflow.MlflowClient()
                    client.log_inputs(run_id=run_id, datasets=[dataset_input])
            except Exception as e:
                _logger.warning('Failed to log evaluation dataset information to MLflow Tracking. Reason: %s', e)
        return predict_result
    else:
        return original(self, *args, **kwargs)