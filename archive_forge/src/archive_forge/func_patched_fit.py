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
def patched_fit(original, self, *args, **kwargs):
    should_log_post_training_metrics = log_post_training_metrics and _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics()
    with _SparkTrainingSession(estimator=self, allow_children=False) as t:
        if t.should_log():
            with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                fit_result = fit_mlflow(original, self, *args, **kwargs)
            if should_log_post_training_metrics and isinstance(fit_result, Model):
                _AUTOLOGGING_METRICS_MANAGER.register_model(fit_result, mlflow.active_run().info.run_id)
            return fit_result
        else:
            return original(self, *args, **kwargs)