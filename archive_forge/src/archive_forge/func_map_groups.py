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
def map_groups(self, fn: UserDefinedFunction[DataBatch, DataBatch], *, compute: Union[str, ComputeStrategy]=None, batch_format: Optional[str]='default', fn_args: Optional[Iterable[Any]]=None, fn_kwargs: Optional[Dict[str, Any]]=None, **ray_remote_args) -> 'Dataset':
    """Apply the given function to each group of records of this dataset.

        While map_groups() is very flexible, note that it comes with downsides:
            * It may be slower than using more specific methods such as min(), max().
            * It requires that each group fits in memory on a single node.

        In general, prefer to use aggregate() instead of map_groups().

        Examples:
            >>> # Return a single record per group (list of multiple records in,
            >>> # list of a single record out).
            >>> import ray
            >>> import pandas as pd
            >>> import numpy as np
            >>> # Get first value per group.
            >>> ds = ray.data.from_items([ # doctest: +SKIP
            ...     {"group": 1, "value": 1},
            ...     {"group": 1, "value": 2},
            ...     {"group": 2, "value": 3},
            ...     {"group": 2, "value": 4}])
            >>> ds.groupby("group").map_groups( # doctest: +SKIP
            ...     lambda g: {"result": np.array([g["value"][0]])})

            >>> # Return multiple records per group (dataframe in, dataframe out).
            >>> df = pd.DataFrame(
            ...     {"A": ["a", "a", "b"], "B": [1, 1, 3], "C": [4, 6, 5]}
            ... )
            >>> ds = ray.data.from_pandas(df) # doctest: +SKIP
            >>> grouped = ds.groupby("A") # doctest: +SKIP
            >>> grouped.map_groups( # doctest: +SKIP
            ...     lambda g: g.apply(
            ...         lambda c: c / g[c.name].sum() if c.name in ["B", "C"] else c
            ...     )
            ... ) # doctest: +SKIP

        Args:
            fn: The function to apply to each group of records, or a class type
                that can be instantiated to create such a callable. It takes as
                input a batch of all records from a single group, and returns a
                batch of zero or more records, similar to map_batches().
            compute: The compute strategy, either "tasks" (default) to use Ray
                tasks, ``ray.data.ActorPoolStrategy(size=n)`` to use a fixed-size actor
                pool, or ``ray.data.ActorPoolStrategy(min_size=m, max_size=n)`` for an
                autoscaling actor pool.
            batch_format: Specify ``"default"`` to use the default block format
                (NumPy), ``"pandas"`` to select ``pandas.DataFrame``, "pyarrow" to
                select ``pyarrow.Table``, or ``"numpy"`` to select
                ``Dict[str, numpy.ndarray]``, or None to return the underlying block
                exactly as is with no additional formatting.
            fn_args: Arguments to `fn`.
            fn_kwargs: Keyword arguments to `fn`.
            ray_remote_args: Additional resource requirements to request from
                ray (e.g., num_gpus=1 to request GPUs for the map tasks).

        Returns:
            The return type is determined by the return type of ``fn``, and the return
            value is combined from results of all groups.
        """
    if self._key is not None:
        sorted_ds = self._dataset.sort(self._key)
    else:
        sorted_ds = self._dataset.repartition(1)

    def get_key_boundaries(block_accessor: BlockAccessor) -> List[int]:
        """Compute block boundaries based on the key(s)"""
        import numpy as np
        keys = block_accessor.to_numpy(self._key)
        if isinstance(keys, dict):
            convert_to_multi_column_sorted_key = np.vectorize(_MultiColumnSortedKey)
            keys: np.ndarray = convert_to_multi_column_sorted_key(*keys.values())
        boundaries = []
        start = 0
        while start < keys.size:
            end = start + np.searchsorted(keys[start:], keys[start], side='right')
            boundaries.append(end)
            start = end
        return boundaries

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
    return sorted_ds.map_batches(group_fn, batch_size=None, compute=compute, batch_format=batch_format, fn_args=fn_args, fn_kwargs=fn_kwargs, **ray_remote_args)