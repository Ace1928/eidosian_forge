from collections import defaultdict
from contextlib import nullcontext
from functools import reduce
import inspect
import json
import os
import re
import operator
import warnings
import pyarrow as pa
from pyarrow._parquet import (ParquetReader, Statistics,  # noqa
from pyarrow.fs import (LocalFileSystem, FileSystem, FileType,
from pyarrow import filesystem as legacyfs
from pyarrow.util import guid, _is_path_like, _stringify_path, _deprecate_api
def write_to_dataset(table, root_path, partition_cols=None, filesystem=None, use_legacy_dataset=None, schema=None, partitioning=None, basename_template=None, use_threads=None, file_visitor=None, existing_data_behavior=None, **kwargs):
    """Wrapper around dataset.write_dataset for writing a Table to
    Parquet format by partitions.
    For each combination of partition columns and values,
    a subdirectories are created in the following
    manner:

    root_dir/
      group1=value1
        group2=value1
          <uuid>.parquet
        group2=value2
          <uuid>.parquet
      group1=valueN
        group2=value1
          <uuid>.parquet
        group2=valueN
          <uuid>.parquet

    Parameters
    ----------
    table : pyarrow.Table
    root_path : str, pathlib.Path
        The root directory of the dataset.
    partition_cols : list,
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    use_legacy_dataset : bool, optional
        Deprecated and has no effect from PyArrow version 15.0.0.
    schema : Schema, optional
        This Schema of the dataset.
    partitioning : Partitioning or list[str], optional
        The partitioning scheme specified with the
        ``pyarrow.dataset.partitioning()`` function or a list of field names.
        When providing a list of field names, you can use
        ``partitioning_flavor`` to drive which partitioning type should be
        used.
    basename_template : str, optional
        A template string used to generate basenames of written data files.
        The token '{i}' will be replaced with an automatically incremented
        integer. If not specified, it defaults to "guid-{i}.parquet".
    use_threads : bool, default True
        Write files in parallel. If enabled, then maximum parallelism will be
        used determined by the number of available CPU cores.
    file_visitor : function
        If set, this function will be called with a WrittenFile instance
        for each file created during the call.  This object will have both
        a path attribute and a metadata attribute.

        The path attribute will be a string containing the path to
        the created file.

        The metadata attribute will be the parquet metadata of the file.
        This metadata will have the file path attribute set and can be used
        to build a _metadata file.  The metadata attribute will be None if
        the format is not parquet.

        Example visitor which simple collects the filenames created::

            visited_paths = []

            def file_visitor(written_file):
                visited_paths.append(written_file.path)

    existing_data_behavior : 'overwrite_or_ignore' | 'error' | 'delete_matching'
        Controls how the dataset will handle data that already exists in
        the destination. The default behaviour is 'overwrite_or_ignore'.

        'overwrite_or_ignore' will ignore any existing data and will
        overwrite files with the same name as an output file.  Other
        existing files will be ignored.  This behavior, in combination
        with a unique basename_template for each write, will allow for
        an append workflow.

        'error' will raise an error if any data exists in the destination.

        'delete_matching' is useful when you are writing a partitioned
        dataset.  The first time each partition directory is encountered
        the entire directory will be deleted.  This allows you to overwrite
        old partitions completely.
    **kwargs : dict,
        Used as additional kwargs for :func:`pyarrow.dataset.write_dataset`
        function for matching kwargs, and remainder to
        :func:`pyarrow.dataset.ParquetFileFormat.make_write_options`.
        See the docstring of :func:`write_table` and
        :func:`pyarrow.dataset.write_dataset` for the available options.
        Using `metadata_collector` in kwargs allows one to collect the
        file metadata instances of dataset pieces. The file paths in the
        ColumnChunkMetaData will be set relative to `root_path`.

    Examples
    --------
    Generate an example PyArrow Table:

    >>> import pyarrow as pa
    >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
    ...                   'n_legs': [2, 2, 4, 4, 5, 100],
    ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
    ...                              "Brittle stars", "Centipede"]})

    and write it to a partitioned dataset:

    >>> import pyarrow.parquet as pq
    >>> pq.write_to_dataset(table, root_path='dataset_name_3',
    ...                     partition_cols=['year'])
    >>> pq.ParquetDataset('dataset_name_3').files
    ['dataset_name_3/year=2019/...-0.parquet', ...

    Write a single Parquet file into the root folder:

    >>> pq.write_to_dataset(table, root_path='dataset_name_4')
    >>> pq.ParquetDataset('dataset_name_4/').files
    ['dataset_name_4/...-0.parquet']
    """
    if use_legacy_dataset is not None:
        warnings.warn("Passing 'use_legacy_dataset' is deprecated as of pyarrow 15.0.0 and will be removed in a future version.", FutureWarning, stacklevel=2)
    metadata_collector = kwargs.pop('metadata_collector', None)
    msg_confl = "The '{1}' argument is not supported. Use only '{0}' instead."
    if partition_cols is not None and partitioning is not None:
        raise ValueError(msg_confl.format('partitioning', 'partition_cols'))
    if metadata_collector is not None and file_visitor is not None:
        raise ValueError(msg_confl.format('file_visitor', 'metadata_collector'))
    import pyarrow.dataset as ds
    write_dataset_kwargs = dict()
    for key in inspect.signature(ds.write_dataset).parameters:
        if key in kwargs:
            write_dataset_kwargs[key] = kwargs.pop(key)
    write_dataset_kwargs['max_rows_per_group'] = kwargs.pop('row_group_size', kwargs.pop('chunk_size', None))
    if metadata_collector is not None:

        def file_visitor(written_file):
            metadata_collector.append(written_file.metadata)
    parquet_format = ds.ParquetFileFormat()
    write_options = parquet_format.make_write_options(**kwargs)
    if filesystem is not None:
        filesystem = _ensure_filesystem(filesystem)
    if partition_cols:
        part_schema = table.select(partition_cols).schema
        partitioning = ds.partitioning(part_schema, flavor='hive')
    if basename_template is None:
        basename_template = guid() + '-{i}.parquet'
    if existing_data_behavior is None:
        existing_data_behavior = 'overwrite_or_ignore'
    ds.write_dataset(table, root_path, filesystem=filesystem, format=parquet_format, file_options=write_options, schema=schema, partitioning=partitioning, use_threads=use_threads, file_visitor=file_visitor, basename_template=basename_template, existing_data_behavior=existing_data_behavior, **write_dataset_kwargs)
    return