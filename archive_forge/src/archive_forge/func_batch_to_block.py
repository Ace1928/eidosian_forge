import collections
import os
import time
from dataclasses import dataclass
from typing import (
import numpy as np
import ray
from ray import DynamicObjectRefGenerator
from ray.data._internal.util import _check_pyarrow_version, _truncated_repr
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI
import psutil
@staticmethod
def batch_to_block(batch: DataBatch) -> Block:
    """Create a block from user-facing data formats."""
    if isinstance(batch, np.ndarray):
        raise ValueError(f"Error validating {_truncated_repr(batch)}: Standalone numpy arrays are not allowed in Ray 2.5. Return a dict of field -> array, e.g., `{{'data': array}}` instead of `array`.")
    elif isinstance(batch, collections.abc.Mapping):
        import pyarrow as pa
        from ray.data._internal.arrow_block import ArrowBlockAccessor
        try:
            return ArrowBlockAccessor.numpy_to_block(batch)
        except (pa.ArrowNotImplementedError, pa.ArrowInvalid, pa.ArrowTypeError):
            import pandas as pd
            return pd.DataFrame(dict(batch))
    return batch