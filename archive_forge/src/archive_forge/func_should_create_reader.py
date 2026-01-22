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
@property
def should_create_reader(self) -> bool:
    has_implemented_get_read_tasks = type(self).get_read_tasks is not Datasource.get_read_tasks
    has_implemented_estimate_inmemory_data_size = type(self).estimate_inmemory_data_size is not Datasource.estimate_inmemory_data_size
    return not has_implemented_get_read_tasks or not has_implemented_estimate_inmemory_data_size