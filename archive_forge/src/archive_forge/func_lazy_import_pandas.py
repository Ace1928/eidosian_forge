import collections
import heapq
from typing import (
import numpy as np
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.table_block import TableBlockAccessor, TableBlockBuilder
from ray.data._internal.util import find_partitions
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.row import TableRow
def lazy_import_pandas():
    global _pandas
    if _pandas is None:
        import pandas
        _pandas = pandas
    return _pandas