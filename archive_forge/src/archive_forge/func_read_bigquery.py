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
def read_bigquery(project_id: str, dataset: Optional[str]=None, query: Optional[str]=None, *, parallelism: int=-1, ray_remote_args: Dict[str, Any]=None) -> Dataset:
    """Create a dataset from BigQuery.

    The data to read from is specified via the ``project_id``, ``dataset``
    and/or ``query`` parameters. The dataset is created from the results of
    executing ``query`` if a query is provided. Otherwise, the entire
    ``dataset`` is read.

    For more information about BigQuery, see the following concepts:

    - Project id: `Creating and Managing Projects <https://cloud.google.com/resource-manager/docs/creating-managing-projects>`_

    - Dataset: `Datasets Intro <https://cloud.google.com/bigquery/docs/datasets-intro>`_

    - Query: `Query Syntax <https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax>`_

    This method uses the BigQuery Storage Read API which reads in parallel,
    with a Ray read task to handle each stream. The number of streams is
    determined by ``parallelism`` which can be requested from this interface
    or automatically chosen if unspecified (see the ``parallelism`` arg below).

    .. warning::
        The maximum query response size is 10GB. For more information, see `BigQuery response too large to return <https://cloud.google.com/knowledge/kb/bigquery-response-too-large-to-return-consider-setting-allowlargeresults-to-true-in-your-job-configuration-000004266>`_.

    Examples:
        .. testcode::
            :skipif: True

            import ray
            # Users will need to authenticate beforehand (https://cloud.google.com/sdk/gcloud/reference/auth/login)
            ds = ray.data.read_bigquery(
                project_id="my_project",
                query="SELECT * FROM `bigquery-public-data.samples.gsod` LIMIT 1000",
            )

    Args:
        project_id: The name of the associated Google Cloud Project that hosts the dataset to read.
            For more information, see `Creating and Managing Projects <https://cloud.google.com/resource-manager/docs/creating-managing-projects>`_.
        dataset: The name of the dataset hosted in BigQuery in the format of ``dataset_id.table_id``.
            Both the dataset_id and table_id must exist otherwise an exception will be raised.
        parallelism: The requested parallelism of the read. If -1, it will be
            automatically chosen based on the available cluster resources and estimated
            in-memory data size.
        ray_remote_args: kwargs passed to ray.remote in the read tasks.

    Returns:
        Dataset producing rows from the results of executing the query (or reading the entire dataset)
        on the specified BigQuery dataset.
    """
    datasource = BigQueryDatasource(project_id=project_id, dataset=dataset, query=query)
    return read_datasource(datasource, parallelism=parallelism, ray_remote_args=ray_remote_args)