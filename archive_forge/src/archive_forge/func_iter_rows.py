import collections
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, TypeVar, Union
import numpy as np
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.block_builder import BlockBuilder
from ray.data._internal.numpy_support import convert_udf_returns_to_numpy, is_array_like
from ray.data._internal.size_estimator import SizeEstimator
from ray.data.block import Block, BlockAccessor
from ray.data.row import TableRow
def iter_rows(self, public_row_format: bool) -> Iterator[Union[Mapping, np.ndarray]]:
    outer = self

    class Iter:

        def __init__(self):
            self._cur = -1

        def __iter__(self):
            return self

        def __next__(self):
            self._cur += 1
            if self._cur < outer.num_rows():
                row = outer._get_row(self._cur)
                if public_row_format and isinstance(row, TableRow):
                    return row.as_pydict()
                else:
                    return row
            raise StopIteration
    return Iter()