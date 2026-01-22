import logging
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import DataType
from mlflow.types.schema import (
def clean_tensor_type(dtype: np.dtype):
    """
    This method strips away the size information stored in flexible datatypes such as np.str_ and
    np.bytes_. Other numpy dtypes are returned unchanged.

    Args:
        dtype: Numpy dtype of a tensor

    Returns:
        dtype: Cleaned numpy dtype
    """
    if not isinstance(dtype, np.dtype):
        raise TypeError(f'Expected `type` to be instance of `{np.dtype}`, received `{dtype.__class__}`')
    if dtype.char == 'U':
        return np.dtype('str')
    elif dtype.char == 'S':
        return np.dtype('bytes')
    return dtype