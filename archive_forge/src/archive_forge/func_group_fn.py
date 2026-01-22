from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from ._internal.table_block import TableBlockAccessor
from ray.data._internal import sort
from ray.data._internal.compute import ComputeStrategy
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators.all_to_all_operator import Aggregate
from ray.data._internal.plan import AllToAllStage
from ray.data._internal.push_based_shuffle import PushBasedShufflePlan
from ray.data._internal.shuffle import ShuffleOp, SimpleShufflePlan
from ray.data._internal.sort import SortKey
from ray.data.aggregate import AggregateFn, Count, Max, Mean, Min, Std, Sum
from ray.data.aggregate._aggregate import _AggregateOnKeyBase
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.dataset import DataBatch, Dataset
from ray.util.annotations import PublicAPI
def group_fn(batch, *args, **kwargs):
    block = BlockAccessor.batch_to_block(batch)
    block_accessor = BlockAccessor.for_block(block)
    if self._key:
        boundaries = get_key_boundaries(block_accessor)
    else:
        boundaries = [block_accessor.num_rows()]
    builder = DelegatingBlockBuilder()
    start = 0
    for end in boundaries:
        group_block = block_accessor.slice(start, end)
        group_block_accessor = BlockAccessor.for_block(group_block)
        group_batch = group_block_accessor.to_batch_format(batch_format)
        applied = fn(group_batch, *args, **kwargs)
        builder.add_batch(applied)
        start = end
    rs = builder.build()
    return rs