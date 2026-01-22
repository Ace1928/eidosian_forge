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
@PublicAPI(stability='alpha')
def read_mongo(uri: str, database: str, collection: str, *, pipeline: Optional[List[Dict]]=None, schema: Optional['pymongoarrow.api.Schema']=None, parallelism: int=-1, ray_remote_args: Dict[str, Any]=None, **mongo_args) -> Dataset:
    """Create a :class:`~ray.data.Dataset` from a MongoDB database.

    The data to read from is specified via the ``uri``, ``database`` and ``collection``
    of the MongoDB. The dataset is created from the results of executing
    ``pipeline`` against the ``collection``. If ``pipeline`` is None, the entire
    ``collection`` is read.

    .. tip::

        For more details about these MongoDB concepts, see the following:
        - URI: https://www.mongodb.com/docs/manual/reference/connection-string/
        - Database and Collection: https://www.mongodb.com/docs/manual/core/databases-and-collections/
        - Pipeline: https://www.mongodb.com/docs/manual/core/aggregation-pipeline/

    To read the MongoDB in parallel, the execution of the pipeline is run on partitions
    of the collection, with a Ray read task to handle a partition. Partitions are
    created in an attempt to evenly distribute the documents into the specified number
    of partitions. The number of partitions is determined by ``parallelism`` which can
    be requested from this interface or automatically chosen if unspecified (see the
    ``parallelism`` arg below).

    Examples:
        >>> import ray
        >>> from pymongoarrow.api import Schema # doctest: +SKIP
        >>> ds = ray.data.read_mongo( # doctest: +SKIP
        ...     uri="mongodb://username:password@mongodb0.example.com:27017/?authSource=admin", # noqa: E501
        ...     database="my_db",
        ...     collection="my_collection",
        ...     pipeline=[{"$match": {"col2": {"$gte": 0, "$lt": 100}}}, {"$sort": "sort_field"}], # noqa: E501
        ...     schema=Schema({"col1": pa.string(), "col2": pa.int64()}),
        ...     parallelism=10,
        ... )

    Args:
        uri: The URI of the source MongoDB where the dataset is
            read from. For the URI format, see details in the `MongoDB docs <https:/                /www.mongodb.com/docs/manual/reference/connection-string/>`_.
        database: The name of the database hosted in the MongoDB. This database
            must exist otherwise ValueError is raised.
        collection: The name of the collection in the database. This collection
            must exist otherwise ValueError is raised.
        pipeline: A `MongoDB pipeline <https://www.mongodb.com/docs/manual/core            /aggregation-pipeline/>`_, which is executed on the given collection
            with results used to create Dataset. If None, the entire collection will
            be read.
        schema: The schema used to read the collection. If None, it'll be inferred from
            the results of pipeline.
        parallelism: The requested parallelism of the read. Defaults to -1,
            which automatically determines the optimal parallelism for your
            configuration. You should not need to manually set this value in most cases.
            For details on how the parallelism is automatically determined and guidance
            on how to tune it, see :ref:`Tuning read parallelism
            <read_parallelism>`.
        ray_remote_args: kwargs passed to :meth:`~ray.remote` in the read tasks.
        mongo_args: kwargs passed to `aggregate_arrow_all() <https://mongo-arrow            .readthedocs.io/en/latest/api/api.html#pymongoarrow.api            aggregate_arrow_all>`_ in pymongoarrow in producing
            Arrow-formatted results.

    Returns:
        :class:`~ray.data.Dataset` producing rows from the results of executing the pipeline on the specified MongoDB collection.

    Raises:
        ValueError: if ``database`` doesn't exist.
        ValueError: if ``collection`` doesn't exist.
    """
    datasource = MongoDatasource(uri=uri, database=database, collection=collection, pipeline=pipeline, schema=schema, **mongo_args)
    return read_datasource(datasource, parallelism=parallelism, ray_remote_args=ray_remote_args)