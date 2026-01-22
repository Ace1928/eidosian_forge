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
@PublicAPI(stability='alpha')
@ConsumptionAPI
def write_mongo(self, uri: str, database: str, collection: str, ray_remote_args: Dict[str, Any]=None) -> None:
    """Writes the :class:`~ray.data.Dataset` to a MongoDB database.

        This method is only supported for datasets convertible to pyarrow tables.

        The number of parallel writes is determined by the number of blocks in the
        dataset. To control the number of number of blocks, call
        :meth:`~ray.data.Dataset.repartition`.

        .. warning::
            This method supports only a subset of the PyArrow's types, due to the
            limitation of pymongoarrow which is used underneath. Writing unsupported
            types fails on type checking. See all the supported types at:
            https://mongo-arrow.readthedocs.io/en/latest/data_types.html.

        .. note::
            The records are inserted into MongoDB as new documents. If a record has
            the _id field, this _id must be non-existent in MongoDB, otherwise the write
            is rejected and fail (hence preexisting documents are protected from
            being mutated). It's fine to not have _id field in record and MongoDB will
            auto generate one at insertion.

        Examples:

            .. testcode::
                :skipif: True

                import ray

                ds = ray.data.range(100)
                ds.write_mongo(
                    uri="mongodb://username:password@mongodb0.example.com:27017/?authSource=admin",
                    database="my_db",
                    collection="my_collection"
                )

        Args:
            uri: The URI to the destination MongoDB where the dataset is
                written to. For the URI format, see details in the
                `MongoDB docs <https://www.mongodb.com/docs/manual/reference                    /connection-string/>`_.
            database: The name of the database. This database must exist otherwise
                a ValueError is raised.
            collection: The name of the collection in the database. This collection
                must exist otherwise a ValueError is raised.
            ray_remote_args: kwargs passed to :meth:`~ray.remote` in the write tasks.

        Raises:
            ValueError: if ``database`` doesn't exist.
            ValueError: if ``collection`` doesn't exist.
        """
    datasink = _MongoDatasink(uri=uri, database=database, collection=collection)
    self.write_datasink(datasink, ray_remote_args=ray_remote_args)