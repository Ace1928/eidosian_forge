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
def register_prediction_result(self, run_id, eval_dataset_name, predict_result):
    """
        Register the relationship
         id(prediction_result) --> (eval_dataset_name, run_id)
        into map `_pred_result_id_to_dataset_name_and_run_id`
        """
    value = (eval_dataset_name, run_id)
    prediction_result_id = id(predict_result)
    self._pred_result_id_to_dataset_name_and_run_id[prediction_result_id] = value

    def clean_id(id_):
        _AUTOLOGGING_METRICS_MANAGER._pred_result_id_to_dataset_name_and_run_id.pop(id_, None)
    weakref.finalize(predict_result, clean_id, prediction_result_id)