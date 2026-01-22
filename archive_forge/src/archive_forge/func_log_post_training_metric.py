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
def log_post_training_metric(self, run_id, key, value):
    """
        Log the metric into the specified mlflow run.
        and it will also update the metric_info artifact if needed.
        """
    client = MlflowClient()
    client.log_metric(run_id=run_id, key=key, value=value)
    if self._metric_info_artifact_need_update[run_id]:
        evaluator_call_list = []
        for v in self._evaluator_call_info[run_id].values():
            evaluator_call_list.extend(v)
        evaluator_call_list.sort(key=lambda x: x[0])
        dict_to_log = OrderedDict(evaluator_call_list)
        client.log_dict(run_id=run_id, dictionary=dict_to_log, artifact_file='metric_info.json')
        self._metric_info_artifact_need_update[run_id] = False