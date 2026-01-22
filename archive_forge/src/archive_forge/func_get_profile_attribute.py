import json
import logging
from functools import cached_property
from typing import Any, Dict, Optional, Union
import numpy as np
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import compute_numpy_digest
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.data.schema import TensorDatasetSchema
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.types.utils import _infer_schema
def get_profile_attribute(numpy_data, attr_name):
    if isinstance(numpy_data, dict):
        return {key: getattr(array, attr_name) for key, array in numpy_data.items()}
    else:
        return getattr(numpy_data, attr_name)