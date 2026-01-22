import json
import logging
import os
import sys
import traceback
import weakref
from collections import OrderedDict, defaultdict, namedtuple
from itertools import zip_longest
from urllib.parse import urlparse
import numpy as np
import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.spark_dataset import SparkDataset
from mlflow.entities import Metric, Param
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.utils import (
from mlflow.utils.autologging_utils import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.os import is_windows
from mlflow.utils.rest_utils import (
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
def patched_evaluate(original, self, *args, **kwargs):
    if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics():
        with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
            metric = original(self, *args, **kwargs)
        if _AUTOLOGGING_METRICS_MANAGER.is_metric_value_loggable(metric):
            params = get_method_call_arg_value(1, 'params', None, args, kwargs)
            evaluator = self.copy(params) if params is not None else self
            metric_name = evaluator.getMetricName()
            evaluator_info = _AUTOLOGGING_METRICS_MANAGER.gen_evaluator_info(evaluator)
            pred_result_dataset = get_method_call_arg_value(0, 'dataset', None, args, kwargs)
            run_id, dataset_name = _AUTOLOGGING_METRICS_MANAGER.get_run_id_and_dataset_name_for_evaluator_call(pred_result_dataset)
            if run_id and dataset_name:
                metric_key = _AUTOLOGGING_METRICS_MANAGER.register_evaluator_call(run_id, metric_name, dataset_name, evaluator_info)
                _AUTOLOGGING_METRICS_MANAGER.log_post_training_metric(run_id, metric_key, metric)
                if log_datasets:
                    try:
                        context_tags = context_registry.resolve_tags()
                        code_source = CodeDatasetSource(context_tags)
                        dataset = SparkDataset(df=pred_result_dataset, source=code_source)
                        tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value='eval')]
                        dataset_input = DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags)
                        client = MlflowClient()
                        client.log_inputs(run_id, [dataset_input])
                    except Exception as e:
                        _logger.warning('Failed to log evaluation dataset information to MLflow Tracking. Reason: %s', e)
        return metric
    else:
        return original(self, *args, **kwargs)