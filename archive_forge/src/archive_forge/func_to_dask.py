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
def to_dask(self, meta: Union['pandas.DataFrame', 'pandas.Series', Dict[str, Any], Iterable[Any], Tuple[Any], None]=None, verify_meta: bool=True) -> 'dask.dataframe.DataFrame':
    """Convert this :class:`~ray.data.Dataset` into a
        `Dask DataFrame <https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.html#dask.dataframe.DataFrame>`_.

        This is only supported for datasets convertible to Arrow records.

        Note that this function will set the Dask scheduler to Dask-on-Ray
        globally, via the config.

        Time complexity: O(dataset size / parallelism)

        Args:
            meta: An empty `pandas DataFrame`_ or `Series`_ that matches the dtypes and column
                names of the stream. This metadata is necessary for many algorithms in
                dask dataframe to work. For ease of use, some alternative inputs are
                also available. Instead of a DataFrame, a dict of ``{name: dtype}`` or
                iterable of ``(name, dtype)`` can be provided (note that the order of
                the names should match the order of the columns). Instead of a series, a
                tuple of ``(name, dtype)`` can be used.
                By default, this is inferred from the underlying Dataset schema,
                with this argument supplying an optional override.
            verify_meta: If True, Dask will check that the partitions have consistent
                metadata. Defaults to True.

        Returns:
            A `Dask DataFrame`_ created from this dataset.

        .. _pandas DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
        .. _Series: https://pandas.pydata.org/docs/reference/api/pandas.Series.html
        """
    import dask
    import dask.dataframe as dd
    import pandas as pd
    try:
        import pyarrow as pa
    except Exception:
        pa = None
    from ray.data._internal.pandas_block import PandasBlockSchema
    from ray.util.client.common import ClientObjectRef
    from ray.util.dask import ray_dask_get
    dask.config.set(scheduler=ray_dask_get)

    @dask.delayed
    def block_to_df(block: Block):
        if isinstance(block, (ray.ObjectRef, ClientObjectRef)):
            raise ValueError('Dataset.to_dask() must be used with Dask-on-Ray, please set the Dask scheduler to ray_dask_get (located in ray.util.dask).')
        return _block_to_df(block)
    if meta is None:
        from ray.data.extensions import TensorDtype
        schema = self.schema(fetch_if_missing=True)
        if isinstance(schema, PandasBlockSchema):
            meta = pd.DataFrame({col: pd.Series(dtype=dtype if not isinstance(dtype, TensorDtype) else np.object_) for col, dtype in zip(schema.names, schema.types)})
        elif pa is not None and isinstance(schema, pa.Schema):
            from ray.data.extensions import ArrowTensorType
            if any((isinstance(type_, ArrowTensorType) for type_ in schema.types)):
                meta = pd.DataFrame({col: pd.Series(dtype=dtype.to_pandas_dtype() if not isinstance(dtype, ArrowTensorType) else np.object_) for col, dtype in zip(schema.names, schema.types)})
            else:
                meta = schema.empty_table().to_pandas()
    ddf = dd.from_delayed([block_to_df(block) for block in self.get_internal_block_refs()], meta=meta, verify_meta=verify_meta)
    return ddf