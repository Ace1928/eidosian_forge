import json
import logging
from functools import cached_property
from typing import Any, Dict, Optional, Union
import pandas as pd
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import compute_pandas_digest
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema
def to_evaluation_dataset(self, path=None, feature_names=None) -> EvaluationDataset:
    """
        Converts the dataset to an EvaluationDataset for model evaluation. Required
        for use with mlflow.evaluate().
        """
    return EvaluationDataset(data=self._df, targets=self._targets, path=path, feature_names=feature_names, predictions=self._predictions)