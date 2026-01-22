import pyarrow as pa
from pyarrow.util import _is_iterable, _stringify_path, _is_path_like
from pyarrow.compute import Expression, scalar, field  # noqa
def write_dataset(data, base_dir, *, basename_template=None, format=None, partitioning=None, partitioning_flavor=None, schema=None, filesystem=None, file_options=None, use_threads=True, max_partitions=None, max_open_files=None, max_rows_per_file=None, min_rows_per_group=None, max_rows_per_group=None, file_visitor=None, existing_data_behavior='error', create_dir=True):
    """
    Write a dataset to a given format and partitioning.

    Parameters
    ----------
    data : Dataset, Table/RecordBatch, RecordBatchReader, list of Table/RecordBatch, or iterable of RecordBatch
        The data to write. This can be a Dataset instance or
        in-memory Arrow data. If an iterable is given, the schema must
        also be given.
    base_dir : str
        The root directory where to write the dataset.
    basename_template : str, optional
        A template string used to generate basenames of written data files.
        The token '{i}' will be replaced with an automatically incremented
        integer. If not specified, it defaults to
        "part-{i}." + format.default_extname
    format : FileFormat or str
        The format in which to write the dataset. Currently supported:
        "parquet", "ipc"/"arrow"/"feather", and "csv". If a FileSystemDataset
        is being written and `format` is not specified, it defaults to the
        same format as the specified FileSystemDataset. When writing a
        Table or RecordBatch, this keyword is required.
    partitioning : Partitioning or list[str], optional
        The partitioning scheme specified with the ``partitioning()``
        function or a list of field names. When providing a list of
        field names, you can use ``partitioning_flavor`` to drive which
        partitioning type should be used.
    partitioning_flavor : str, optional
        One of the partitioning flavors supported by
        ``pyarrow.dataset.partitioning``. If omitted will use the
        default of ``partitioning()`` which is directory partitioning.
    schema : Schema, optional
    filesystem : FileSystem, optional
    file_options : pyarrow.dataset.FileWriteOptions, optional
        FileFormat specific write options, created using the
        ``FileFormat.make_write_options()`` function.
    use_threads : bool, default True
        Write files in parallel. If enabled, then maximum parallelism will be
        used determined by the number of available CPU cores.
    max_partitions : int, default 1024
        Maximum number of partitions any batch may be written into.
    max_open_files : int, default 1024
        If greater than 0 then this will limit the maximum number of
        files that can be left open. If an attempt is made to open
        too many files then the least recently used file will be closed.
        If this setting is set too low you may end up fragmenting your
        data into many small files.
    max_rows_per_file : int, default 0
        Maximum number of rows per file. If greater than 0 then this will
        limit how many rows are placed in any single file. Otherwise there
        will be no limit and one file will be created in each output
        directory unless files need to be closed to respect max_open_files
    min_rows_per_group : int, default 0
        Minimum number of rows per group. When the value is greater than 0,
        the dataset writer will batch incoming data and only write the row
        groups to the disk when sufficient rows have accumulated.
    max_rows_per_group : int, default 1024 * 1024
        Maximum number of rows per group. If the value is greater than 0,
        then the dataset writer may split up large incoming batches into
        multiple row groups.  If this value is set, then min_rows_per_group
        should also be set. Otherwise it could end up with very small row
        groups.
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
    existing_data_behavior : 'error' | 'overwrite_or_ignore' | 'delete_matching'
        Controls how the dataset will handle data that already exists in
        the destination.  The default behavior ('error') is to raise an error
        if any data exists in the destination.

        'overwrite_or_ignore' will ignore any existing data and will
        overwrite files with the same name as an output file.  Other
        existing files will be ignored.  This behavior, in combination
        with a unique basename_template for each write, will allow for
        an append workflow.

        'delete_matching' is useful when you are writing a partitioned
        dataset.  The first time each partition directory is encountered
        the entire directory will be deleted.  This allows you to overwrite
        old partitions completely.
    create_dir : bool, default True
        If False, directories will not be created.  This can be useful for
        filesystems that do not require directories.
    """
    from pyarrow.fs import _resolve_filesystem_and_path
    if isinstance(data, (list, tuple)):
        schema = schema or data[0].schema
        data = InMemoryDataset(data, schema=schema)
    elif isinstance(data, (pa.RecordBatch, pa.Table)):
        schema = schema or data.schema
        data = InMemoryDataset(data, schema=schema)
    elif isinstance(data, pa.ipc.RecordBatchReader) or _is_iterable(data):
        data = Scanner.from_batches(data, schema=schema)
        schema = None
    elif not isinstance(data, (Dataset, Scanner)):
        raise ValueError('Only Dataset, Scanner, Table/RecordBatch, RecordBatchReader, a list of Tables/RecordBatches, or iterable of batches are supported.')
    if format is None and isinstance(data, FileSystemDataset):
        format = data.format
    else:
        format = _ensure_format(format)
    if file_options is None:
        file_options = format.make_write_options()
    if format != file_options.format:
        raise TypeError("Supplied FileWriteOptions have format {}, which doesn't match supplied FileFormat {}".format(format, file_options))
    if basename_template is None:
        basename_template = 'part-{i}.' + format.default_extname
    if max_partitions is None:
        max_partitions = 1024
    if max_open_files is None:
        max_open_files = 1024
    if max_rows_per_file is None:
        max_rows_per_file = 0
    if max_rows_per_group is None:
        max_rows_per_group = 1 << 20
    if min_rows_per_group is None:
        min_rows_per_group = 0
    if isinstance(data, Scanner):
        partitioning_schema = data.projected_schema
    else:
        partitioning_schema = data.schema
    partitioning = _ensure_write_partitioning(partitioning, schema=partitioning_schema, flavor=partitioning_flavor)
    filesystem, base_dir = _resolve_filesystem_and_path(base_dir, filesystem)
    if isinstance(data, Dataset):
        scanner = data.scanner(use_threads=use_threads)
    else:
        if schema is not None:
            raise ValueError('Cannot specify a schema when writing a Scanner')
        scanner = data
    _filesystemdataset_write(scanner, base_dir, basename_template, filesystem, partitioning, file_options, max_partitions, file_visitor, existing_data_behavior, max_open_files, max_rows_per_file, min_rows_per_group, max_rows_per_group, create_dir)