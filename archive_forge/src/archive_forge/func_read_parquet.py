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
def read_parquet(paths: Union[str, List[str]], *, filesystem: Optional['pyarrow.fs.FileSystem']=None, columns: Optional[List[str]]=None, parallelism: int=-1, ray_remote_args: Dict[str, Any]=None, tensor_column_schema: Optional[Dict[str, Tuple[np.dtype, Tuple[int, ...]]]]=None, meta_provider: Optional[ParquetMetadataProvider]=None, partition_filter: Optional[PathPartitionFilter]=None, shuffle: Union[Literal['files'], None]=None, include_paths: bool=False, file_extensions: Optional[List[str]]=None, **arrow_parquet_args) -> Dataset:
    """Creates a :class:`~ray.data.Dataset` from parquet files.


    Examples:
        Read a file in remote storage.

        >>> import ray
        >>> ds = ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet")
        >>> ds.schema()
        Column        Type
        ------        ----
        sepal.length  double
        sepal.width   double
        petal.length  double
        petal.width   double
        variety       string

        Read a directory in remote storage.

        >>> ds = ray.data.read_parquet("s3://anonymous@ray-example-data/iris-parquet/")

        Read multiple local files.

        >>> ray.data.read_parquet(
        ...    ["local:///path/to/file1", "local:///path/to/file2"]) # doctest: +SKIP

        Specify a schema for the parquet file.

        >>> import pyarrow as pa
        >>> fields = [("sepal.length", pa.float32()),
        ...           ("sepal.width", pa.float32()),
        ...           ("petal.length", pa.float32()),
        ...           ("petal.width", pa.float32()),
        ...           ("variety", pa.string())]
        >>> ds = ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet",
        ...     schema=pa.schema(fields))
        >>> ds.schema()
        Column        Type
        ------        ----
        sepal.length  float
        sepal.width   float
        petal.length  float
        petal.width   float
        variety       string

        The Parquet reader also supports projection and filter pushdown, allowing column
        selection and row filtering to be pushed down to the file scan.

        .. testcode::

            import pyarrow as pa

            # Create a Dataset by reading a Parquet file, pushing column selection and
            # row filtering down to the file scan.
            ds = ray.data.read_parquet(
                "s3://anonymous@ray-example-data/iris.parquet",
                columns=["sepal.length", "variety"],
                filter=pa.dataset.field("sepal.length") > 5.0,
            )

            ds.show(2)

        .. testoutput::

            {'sepal.length': 5.1, 'variety': 'Setosa'}
            {'sepal.length': 5.4, 'variety': 'Setosa'}

        For further arguments you can pass to PyArrow as a keyword argument, see the
        `PyArrow API reference <https://arrow.apache.org/docs/python/generated/        pyarrow.dataset.Scanner.html#pyarrow.dataset.Scanner.from_fragment>`_.

    Args:
        paths: A single file path or directory, or a list of file paths. Multiple
            directories are not supported.
        filesystem: The PyArrow filesystem
            implementation to read from. These filesystems are specified in the
            `pyarrow docs <https://arrow.apache.org/docs/python/api/            filesystems.html#filesystem-implementations>`_. Specify this parameter if
            you need to provide specific configurations to the filesystem. By default,
            the filesystem is automatically selected based on the scheme of the paths.
            For example, if the path begins with ``s3://``, the ``S3FileSystem`` is
            used. If ``None``, this function uses a system-chosen implementation.
        columns: A list of column names to read. Only the specified columns are
            read during the file scan.
        parallelism: The amount of parallelism to use for the dataset. Defaults to -1,
            which automatically determines the optimal parallelism for your
            configuration. You should not need to manually set this value in most cases.
            For details on how the parallelism is automatically determined and guidance
            on how to tune it, see :ref:`Tuning read parallelism
            <read_parallelism>`. Parallelism is upper bounded by the total number of
            records in all the parquet files.
        ray_remote_args: kwargs passed to :meth:`~ray.remote` in the read tasks.
        tensor_column_schema: A dict of column name to PyArrow dtype and shape
            mappings for converting a Parquet column containing serialized
            tensors (ndarrays) as their elements to PyArrow tensors. This function
            assumes that the tensors are serialized in the raw
            NumPy array format in C-contiguous order (e.g., via
            `arr.tobytes()`).
        meta_provider: A :ref:`file metadata provider <metadata_provider>`. Custom
            metadata providers may be able to resolve file metadata more quickly and/or
            accurately. In most cases you do not need to set this parameter.
        partition_filter: A
            :class:`~ray.data.datasource.partitioning.PathPartitionFilter`. Use
            with a custom callback to read only selected partitions of a dataset.
        shuffle: If setting to "files", randomly shuffle input files order before read.
            Defaults to not shuffle with ``None``.
        arrow_parquet_args: Other parquet read options to pass to PyArrow. For the full
            set of arguments, see the`PyArrow API <https://arrow.apache.org/docs/                python/generated/pyarrow.dataset.Scanner.html                    #pyarrow.dataset.Scanner.from_fragment>`_
        include_paths: If ``True``, include the path to each file. File paths are
            stored in the ``'path'`` column.
        file_extensions: A list of file extensions to filter files by.

    Returns:
        :class:`~ray.data.Dataset` producing records read from the specified parquet
        files.
    """
    if meta_provider is None:
        meta_provider = get_parquet_metadata_provider()
    arrow_parquet_args = _resolve_parquet_args(tensor_column_schema, **arrow_parquet_args)
    dataset_kwargs = arrow_parquet_args.pop('dataset_kwargs', None)
    _block_udf = arrow_parquet_args.pop('_block_udf', None)
    schema = arrow_parquet_args.pop('schema', None)
    datasource = ParquetDatasource(paths, columns=columns, dataset_kwargs=dataset_kwargs, to_batch_kwargs=arrow_parquet_args, _block_udf=_block_udf, filesystem=filesystem, schema=schema, meta_provider=meta_provider, partition_filter=partition_filter, shuffle=shuffle, include_paths=include_paths, file_extensions=file_extensions)
    return read_datasource(datasource, parallelism=parallelism, ray_remote_args=ray_remote_args)