import collections
from types import GeneratorType
from typing import Any, Callable, Iterable, Iterator, Optional
import numpy as np
import pandas as pd
import pyarrow as pa
import ray
from ray.data._internal.compute import get_compute
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.execution.util import make_callable_class_concurrent
from ray.data._internal.logical.operators.map_operator import (
from ray.data._internal.numpy_support import is_valid_udf_return
from ray.data._internal.util import _truncated_repr
from ray.data.block import (
from ray.data.context import DataContext
def plan_udf_map_op(op: AbstractUDFMap, input_physical_dag: PhysicalOperator) -> MapOperator:
    """Get the corresponding physical operators DAG for AbstractUDFMap operators.

    Note this method only converts the given `op`, but not its input dependencies.
    See Planner.plan() for more details.
    """
    compute = get_compute(op._compute)
    fn, init_fn = _parse_op_fn(op)
    if isinstance(op, MapBatches):
        transform_fn = _generate_transform_fn_for_map_batches(fn)
        map_transformer = _create_map_transformer_for_map_batches_op(transform_fn, op._batch_size, op._batch_format, op._zero_copy_batch, init_fn)
    else:
        if isinstance(op, MapRows):
            transform_fn = _generate_transform_fn_for_map_rows(fn)
        elif isinstance(op, FlatMap):
            transform_fn = _generate_transform_fn_for_flat_map(fn)
        elif isinstance(op, Filter):
            transform_fn = _generate_transform_fn_for_filter(fn)
        else:
            raise ValueError(f'Found unknown logical operator during planning: {op}')
        map_transformer = _create_map_transformer_for_row_based_map_op(transform_fn, init_fn)
    return MapOperator.create(map_transformer, input_physical_dag, name=op.name, target_max_block_size=None, compute_strategy=compute, min_rows_per_bundle=op._min_rows_per_block, ray_remote_args=op._ray_remote_args)