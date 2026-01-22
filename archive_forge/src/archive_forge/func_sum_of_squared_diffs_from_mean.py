import collections
import heapq
import random
from typing import (
import numpy as np
from ray._private.utils import _get_pyarrow_version
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.arrow_ops import transform_polars, transform_pyarrow
from ray.data._internal.numpy_support import (
from ray.data._internal.table_block import TableBlockAccessor, TableBlockBuilder
from ray.data._internal.util import _truncated_repr, find_partitions
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.row import TableRow
def sum_of_squared_diffs_from_mean(self, on: str, ignore_nulls: bool, mean: Optional[U]=None) -> Optional[U]:
    import pyarrow.compute as pac
    if mean is None:
        mean = self.mean(on, ignore_nulls)
        if mean is None:
            return None
    return self._apply_arrow_compute(lambda col, skip_nulls: pac.sum(pac.power(pac.subtract(col, mean), 2), skip_nulls=skip_nulls), on, ignore_nulls)