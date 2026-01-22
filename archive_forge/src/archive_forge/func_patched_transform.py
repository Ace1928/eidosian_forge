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
def patched_transform(original, self, *args, **kwargs):
    run_id = _AUTOLOGGING_METRICS_MANAGER.get_run_id_for_model(self)
    if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics() and run_id:
        predict_result = original(self, *args, **kwargs)
        eval_dataset = get_method_call_arg_value(0, 'dataset', None, args, kwargs)
        eval_dataset_name = _AUTOLOGGING_METRICS_MANAGER.register_prediction_input_dataset(self, eval_dataset)
        _AUTOLOGGING_METRICS_MANAGER.register_prediction_result(run_id, eval_dataset_name, predict_result)
        return predict_result
    else:
        return original(self, *args, **kwargs)