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
def register_evaluator_call(self, run_id, metric_name, dataset_name, evaluator_info):
    """
        Register the `Evaluator.evaluate` call, including register the evaluator information
        (See doc of `gen_evaluator_info` method) into the corresponding run_id and metric_name
        entry in the registry table.
        """
    evaluator_call_info_list = self._evaluator_call_info[run_id][metric_name]
    index = len(evaluator_call_info_list)
    metric_name_with_index = self.gen_name_with_index(metric_name, index)
    metric_key = f'{metric_name_with_index}_{dataset_name}'
    evaluator_call_info_list.append((metric_key, evaluator_info))
    self._metric_info_artifact_need_update[run_id] = True
    return metric_key