import collections
import logging
import os
from typing import (
import numpy as np
import ray
from ray._private.auto_init_hook import wrap_auto_init
from ray.air.util.tensor_extensions.utils import _create_possibly_ragged_ndarray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.from_operators import (
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.optimizers import LogicalPlan
from ray.data._internal.plan import ExecutionPlan
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import DatasetStats
from ray.data._internal.util import (
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.data.dataset import Dataset, MaterializedDataset
from ray.data.datasource import (
from ray.data.datasource._default_metadata_providers import (
from ray.data.datasource.datasource import Reader
from ray.data.datasource.file_based_datasource import (
from ray.data.datasource.partitioning import Partitioning
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
@PublicAPI
def range_tensor(n: int, *, shape: Tuple=(1,), parallelism: int=-1) -> Dataset:
    """Creates a :class:`~ray.data.Dataset` tensors of the provided shape from range
    [0...n].

    This function allows for easy creation of synthetic tensor datasets for testing or
    benchmarking :ref:`Ray Data <data>`.

    Examples:

        >>> import ray
        >>> ds = ray.data.range_tensor(1000, shape=(2, 2))
        >>> ds
        Dataset(
           num_blocks=...,
           num_rows=1000,
           schema={data: numpy.ndarray(shape=(2, 2), dtype=int64)}
        )
        >>> ds.map_batches(lambda row: {"data": row["data"] * 2}).take(2)
        [{'data': array([[0, 0],
               [0, 0]])}, {'data': array([[2, 2],
               [2, 2]])}]

    Args:
        n: The upper bound of the range of tensor records.
        shape: The shape of each tensor in the dataset.
        parallelism: The amount of parallelism to use for the dataset. Defaults to -1,
            which automatically determines the optimal parallelism for your
            configuration. You should not need to manually set this value in most cases.
            For details on how the parallelism is automatically determined and guidance
            on how to tune it, see
            :ref:`Tuning read parallelism <read_parallelism>`.
            Parallelism is upper bounded by n.

    Returns:
        A :class:`~ray.data.Dataset` producing the tensor data from range 0 to n.

    .. seealso::

        :meth:`~ray.data.range`
                    Call this method to create synthetic datasets of integer data.

    """
    datasource = RangeDatasource(n=n, block_format='tensor', column_name='data', tensor_shape=tuple(shape))
    return read_datasource(datasource, parallelism=parallelism)