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
def do_agg(blocks, task_ctx: TaskContext, clear_input_blocks: bool, *_):
    stage_info = {}
    if len(aggs) == 0:
        raise ValueError('Aggregate requires at least one aggregation')
    for agg in aggs:
        agg._validate(self._dataset.schema(fetch_if_missing=True))
    if blocks.initial_num_blocks() == 0:
        return (blocks, stage_info)
    num_mappers = blocks.initial_num_blocks()
    num_reducers = num_mappers
    if self._key is None:
        num_reducers = 1
        boundaries = []
    else:
        boundaries = sort.sample_boundaries(blocks.get_blocks(), SortKey(self._key), num_reducers, task_ctx)
    ctx = DataContext.get_current()
    if ctx.use_push_based_shuffle:
        shuffle_op_cls = PushBasedGroupbyOp
    else:
        shuffle_op_cls = SimpleShuffleGroupbyOp
    shuffle_op = shuffle_op_cls(map_args=[boundaries, self._key, aggs], reduce_args=[self._key, aggs])
    return shuffle_op.execute(blocks, num_reducers, clear_input_blocks, ctx=task_ctx)