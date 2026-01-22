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
def get_run_id_and_dataset_name_for_evaluator_call(self, pred_result_dataset):
    """
        Given a registered prediction result dataset object,
        return a tuple of (run_id, eval_dataset_name)
        """
    if id(pred_result_dataset) in self._pred_result_id_to_dataset_name_and_run_id:
        dataset_name, run_id = self._pred_result_id_to_dataset_name_and_run_id[id(pred_result_dataset)]
        return (run_id, dataset_name)
    else:
        return (None, None)