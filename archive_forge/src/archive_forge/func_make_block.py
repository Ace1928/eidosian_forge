import warnings
from typing import Any, Callable, Iterable, List, Optional
import numpy as np
import ray
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.util import _check_pyarrow_version
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
def make_block(count: int, num_columns: int) -> Block:
    return pyarrow.Table.from_arrays(np.random.randint(np.iinfo(np.int64).max, size=(num_columns, count), dtype=np.int64), names=[f'c_{i}' for i in range(num_columns)])