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
def iter_tf_batches(self, *, prefetch_batches: int=1, batch_size: Optional[int]=256, dtypes: Optional[Union['tf.dtypes.DType', Dict[str, 'tf.dtypes.DType']]]=None, drop_last: bool=False, local_shuffle_buffer_size: Optional[int]=None, local_shuffle_seed: Optional[int]=None, prefetch_blocks: int=0) -> Iterable[TensorFlowTensorBatchType]:
    """Return an iterable over batches of data represented as TensorFlow tensors.

        This iterable yields batches of type ``Dict[str, tf.Tensor]``.
        For more flexibility, call :meth:`~Dataset.iter_batches` and manually convert
        your data to TensorFlow tensors.

        .. tip::
            If you don't need the additional flexibility provided by this method,
            consider using :meth:`~ray.data.Dataset.to_tf` instead. It's easier
            to use.

        Examples:

            .. testcode::

                import ray

                ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

                tf_dataset = ds.to_tf(
                    feature_columns="sepal length (cm)",
                    label_columns="target",
                    batch_size=2
                )
                for features, labels in tf_dataset:
                    print(features, labels)

            .. testoutput::

                tf.Tensor([5.1 4.9], shape=(2,), dtype=float64) tf.Tensor([0 0], shape=(2,), dtype=int64)
                ...
                tf.Tensor([6.2 5.9], shape=(2,), dtype=float64) tf.Tensor([2 2], shape=(2,), dtype=int64)

        Time complexity: O(1)

        Args:
            prefetch_batches: The number of batches to fetch ahead of the current batch
                to fetch. If set to greater than 0, a separate threadpool is used
                to fetch the objects to the local node, format the batches, and apply
                the ``collate_fn``. Defaults to 1.
            batch_size: The number of rows in each batch, or ``None`` to use entire
                blocks as batches (blocks may contain different numbers of rows).
                The final batch may include fewer than ``batch_size`` rows if
                ``drop_last`` is ``False``. Defaults to 256.
            dtypes: The TensorFlow dtype(s) for the created tensor(s); if ``None``, the
                dtype is inferred from the tensor data.
            drop_last: Whether to drop the last batch if it's incomplete.
            local_shuffle_buffer_size: If not ``None``, the data is randomly shuffled
                using a local in-memory shuffle buffer, and this value serves as the
                minimum number of rows that must be in the local in-memory shuffle
                buffer in order to yield a batch. When there are no more rows to add to
                the buffer, the remaining rows in the buffer are drained.
                ``batch_size`` must also be specified when using local shuffling.
            local_shuffle_seed: The seed to use for the local random shuffle.

        Returns:
            An iterable over TensorFlow Tensor batches.

        .. seealso::
            :meth:`Dataset.iter_batches`
                Call this method to manually convert your data to TensorFlow tensors.
        """
    return self.iterator().iter_tf_batches(prefetch_batches=prefetch_batches, prefetch_blocks=prefetch_blocks, batch_size=batch_size, dtypes=dtypes, drop_last=drop_last, local_shuffle_buffer_size=local_shuffle_buffer_size, local_shuffle_seed=local_shuffle_seed)