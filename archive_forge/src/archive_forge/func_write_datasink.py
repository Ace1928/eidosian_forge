import collections
import copy
import html
import itertools
import logging
import time
import warnings
from typing import (
import numpy as np
import ray
import ray.cloudpickle as pickle
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray._private.usage import usage_lib
from ray.air.util.tensor_extensions.utils import _create_possibly_ragged_ndarray
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import ComputeStrategy, TaskPoolStrategy
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.equalize import _equalize
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.execution.legacy_compat import _block_list_to_bundles
from ray.data._internal.iterator.iterator_impl import DataIteratorImpl
from ray.data._internal.iterator.stream_split_iterator import StreamSplitDataIterator
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.all_to_all_operator import (
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.map_operator import (
from ray.data._internal.logical.operators.n_ary_operator import (
from ray.data._internal.logical.operators.n_ary_operator import Zip
from ray.data._internal.logical.operators.one_to_one_operator import Limit
from ray.data._internal.logical.operators.write_operator import Write
from ray.data._internal.logical.optimizers import LogicalPlan
from ray.data._internal.pandas_block import PandasBlockSchema
from ray.data._internal.plan import ExecutionPlan, OneToOneStage
from ray.data._internal.planner.plan_udf_map_op import (
from ray.data._internal.planner.plan_write_op import generate_write_fn
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.sort import SortKey
from ray.data._internal.split import _get_num_rows, _split_at_indices
from ray.data._internal.stage_impl import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary, StatsManager
from ray.data._internal.util import (
from ray.data.aggregate import AggregateFn, Max, Mean, Min, Std, Sum
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.datasource import (
from ray.data.iterator import DataIterator
from ray.data.random_access_dataset import RandomAccessDataset
from ray.types import ObjectRef
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.widgets import Template
from ray.widgets.util import repr_with_fallback
@ConsumptionAPI(pattern='Time complexity:')
def write_datasink(self, datasink: Datasink, *, ray_remote_args: Dict[str, Any]=None) -> None:
    """Writes the dataset to a custom :class:`~ray.data.Datasink`.

        Time complexity: O(dataset size / parallelism)

        Args:
            datasink: The :class:`~ray.data.Datasink` to write to.
            ray_remote_args: Kwargs passed to ``ray.remote`` in the write tasks.
        """
    if ray_remote_args is None:
        ray_remote_args = {}
    if not datasink.supports_distributed_writes:
        if ray.util.client.ray.is_connected():
            raise ValueError("If you're using Ray Client, Ray Data won't schedule write tasks on the driver's node.")
        ray_remote_args['scheduling_strategy'] = NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False)
    write_fn = generate_write_fn(datasink)

    def write_fn_wrapper(blocks: Iterator[Block], ctx, fn) -> Iterator[Block]:
        return write_fn(blocks, ctx)
    plan = self._plan.with_stage(OneToOneStage('Write', write_fn_wrapper, TaskPoolStrategy(), ray_remote_args, fn=lambda x: x))
    logical_plan = self._logical_plan
    if logical_plan is not None:
        write_op = Write(logical_plan.dag, datasink, ray_remote_args=ray_remote_args)
        logical_plan = LogicalPlan(write_op)
    try:
        import pandas as pd
        datasink.on_write_start()
        self._write_ds = Dataset(plan, logical_plan).materialize()
        blocks = ray.get(self._write_ds._plan.execute().get_blocks())
        assert all((isinstance(block, pd.DataFrame) and len(block) == 1 for block in blocks))
        write_results = [block['write_result'][0] for block in blocks]
        datasink.on_write_complete(write_results)
    except Exception as e:
        datasink.on_write_failed(e)
        raise