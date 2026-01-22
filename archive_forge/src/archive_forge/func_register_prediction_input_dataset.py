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
def register_prediction_input_dataset(self, model, eval_dataset):
    """
        Register prediction input dataset into eval_dataset_info_map, it will do:
         1. inspect eval dataset var name.
         2. check whether eval_dataset_info_map already registered this eval dataset.
            will check by object id.
         3. register eval dataset with id.
         4. return eval dataset name with index.

        Note: this method include inspecting argument variable name.
         So should be called directly from the "patched method", to ensure it capture
         correct argument variable name.
        """
    eval_dataset_name = _inspect_original_var_name(eval_dataset, fallback_name='unknown_dataset')
    eval_dataset_id = id(eval_dataset)
    run_id = self.get_run_id_for_model(model)
    registered_dataset_list = self._eval_dataset_info_map[run_id][eval_dataset_name]
    for i, id_i in enumerate(registered_dataset_list):
        if eval_dataset_id == id_i:
            index = i
            break
    else:
        index = len(registered_dataset_list)
    if index == len(registered_dataset_list):
        registered_dataset_list.append(eval_dataset_id)
    return self.gen_name_with_index(eval_dataset_name, index)