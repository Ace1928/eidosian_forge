import copy
import functools
import inspect
import json
import logging
import math
import pathlib
import pickle
import shutil
import tempfile
import time
import traceback
import warnings
from collections import namedtuple
from functools import partial
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn import metrics as sk_metrics
from sklearn.pipeline import Pipeline as sk_Pipeline
import mlflow
from mlflow import MlflowClient
from mlflow.entities.metric import Metric
from mlflow.exceptions import MlflowException
from mlflow.metrics import (
from mlflow.models.evaluation.artifacts import (
from mlflow.models.evaluation.base import (
from mlflow.models.utils import plot_lines
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.pyfunc import _ServedPyFuncModel
from mlflow.sklearn import _SklearnModelWrapper
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.time import get_current_time_millis
def predict_with_latency(X_copy):
    y_pred_list = []
    pred_latencies = []
    if len(X_copy) == 0:
        raise ValueError('Empty input data')
    is_dataframe = isinstance(X_copy, pd.DataFrame)
    for row in X_copy.iterrows() if is_dataframe else enumerate(X_copy):
        i, row_data = row
        single_input = row_data.to_frame().T if is_dataframe else row_data
        start_time = time.time()
        y_pred = self.model.predict(single_input)
        end_time = time.time()
        pred_latencies.append(end_time - start_time)
        y_pred_list.append(y_pred)
    self.metrics_values.update({_LATENCY_METRIC_NAME: MetricValue(scores=pred_latencies)})
    sample_pred = y_pred_list[0]
    if isinstance(sample_pred, pd.DataFrame):
        return pd.concat(y_pred_list)
    elif isinstance(sample_pred, np.ndarray):
        return np.concatenate(y_pred_list, axis=0)
    elif isinstance(sample_pred, list):
        return sum(y_pred_list, [])
    elif isinstance(sample_pred, pd.Series):
        return pd.concat(y_pred_list, ignore_index=True)
    else:
        raise MlflowException(message=f'Unsupported prediction type {type(sample_pred)} for model type {self.model_type}.', error_code=INVALID_PARAMETER_VALUE)