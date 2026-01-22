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
def iter_groups() -> Iterator[Tuple[KeyType, Block]]:
    """Creates an iterator over zero-copy group views."""
    if key is None:
        yield (None, self.to_block())
        return
    start = end = 0
    iter = self.iter_rows(public_row_format=False)
    next_row = None
    while True:
        try:
            if next_row is None:
                next_row = next(iter)
            next_key = next_row[key]
            while next_row[key] == next_key:
                end += 1
                try:
                    next_row = next(iter)
                except StopIteration:
                    next_row = None
                    break
            yield (next_key, self.slice(start, end))
            start = end
        except StopIteration:
            break