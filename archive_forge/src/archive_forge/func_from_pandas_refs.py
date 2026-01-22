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
@DeveloperAPI
def from_pandas_refs(dfs: Union[ObjectRef['pandas.DataFrame'], List[ObjectRef['pandas.DataFrame']]]) -> MaterializedDataset:
    """Create a :class:`~ray.data.Dataset` from a list of Ray object references to
    pandas dataframes.

    Examples:
        >>> import pandas as pd
        >>> import ray
        >>> df_ref = ray.put(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        >>> ray.data.from_pandas_refs(df_ref)
        MaterializedDataset(num_blocks=1, num_rows=3, schema={a: int64, b: int64})

        Create a Ray Dataset from a list of Pandas Dataframes references.

        >>> ray.data.from_pandas_refs([df_ref, df_ref])
        MaterializedDataset(num_blocks=2, num_rows=6, schema={a: int64, b: int64})

    Args:
        dfs: A Ray object reference to a pandas dataframe, or a list of
             Ray object references to pandas dataframes.

    Returns:
        :class:`~ray.data.Dataset` holding data read from the dataframes.
    """
    if isinstance(dfs, ray.ObjectRef):
        dfs = [dfs]
    elif isinstance(dfs, list):
        for df in dfs:
            if not isinstance(df, ray.ObjectRef):
                raise ValueError(f'Expected list of Ray object refs, got list containing {type(df)}')
    else:
        raise ValueError(f'Expected Ray object ref or list of Ray object refs, got {type(df)}')
    context = DataContext.get_current()
    if context.enable_pandas_block:
        get_metadata = cached_remote_fn(get_table_block_metadata)
        metadata = ray.get([get_metadata.remote(df) for df in dfs])
        logical_plan = LogicalPlan(FromPandas(dfs, metadata))
        return MaterializedDataset(ExecutionPlan(BlockList(dfs, metadata, owned_by_consumer=False), DatasetStats(stages={'FromPandas': metadata}, parent=None), run_by_consumer=False), logical_plan)
    df_to_block = cached_remote_fn(pandas_df_to_arrow_block, num_returns=2)
    res = [df_to_block.remote(df) for df in dfs]
    blocks, metadata = map(list, zip(*res))
    metadata = ray.get(metadata)
    logical_plan = LogicalPlan(FromPandas(blocks, metadata))
    return MaterializedDataset(ExecutionPlan(BlockList(blocks, metadata, owned_by_consumer=False), DatasetStats(stages={'FromPandas': metadata}, parent=None), run_by_consumer=False), logical_plan)