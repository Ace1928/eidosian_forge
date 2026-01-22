from typing import Any
import numpy as np
from ray.air.util.tensor_extensions.utils import create_ragged_ndarray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.util import _truncated_repr
def is_array_like(value: Any) -> bool:
    """Checks whether objects are array-like, excluding numpy scalars."""
    return hasattr(value, '__array__') and hasattr(value, '__len__')