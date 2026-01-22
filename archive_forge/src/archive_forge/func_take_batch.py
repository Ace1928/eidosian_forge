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
@ConsumptionAPI
def take_batch(self, batch_size: int=20, *, batch_format: Optional[str]='default') -> DataBatch:
    """Return up to ``batch_size`` rows from the :class:`Dataset` in a batch.

        Ray Data represents batches as NumPy arrays or pandas DataFrames. You can
        configure the batch type by specifying ``batch_format``.

        This method is useful for inspecting inputs to :meth:`~Dataset.map_batches`.

        .. warning::

            :meth:`~Dataset.take_batch` moves up to ``batch_size`` rows to the caller's
            machine. If ``batch_size`` is large, this method can cause an `
            ``OutOfMemory`` error on the caller.

        Examples:

            >>> import ray
            >>> ds = ray.data.range(100)
            >>> ds.take_batch(5)
            {'id': array([0, 1, 2, 3, 4])}

        Time complexity: O(batch_size specified)

        Args:
            batch_size: The maximum number of rows to return.
            batch_format: If ``"default"`` or ``"numpy"``, batches are
                ``Dict[str, numpy.ndarray]``. If ``"pandas"``, batches are
                ``pandas.DataFrame``.

        Returns:
            A batch of up to ``batch_size`` rows from the dataset.

        Raises:
            ``ValueError``: if the dataset is empty.
        """
    batch_format = _apply_strict_mode_batch_format(batch_format)
    limited_ds = self.limit(batch_size)
    try:
        res = next(iter(limited_ds.iter_batches(batch_size=batch_size, prefetch_batches=0, batch_format=batch_format)))
    except StopIteration:
        raise ValueError('The dataset is empty.')
    self._synchronize_progress_bar()
    self._plan._snapshot_stats = limited_ds._plan.stats()
    return res