from pyarrow.util import _is_path_like, _stringify_path
from pyarrow._fs import (  # noqa
def copy_files(source, destination, source_filesystem=None, destination_filesystem=None, *, chunk_size=1024 * 1024, use_threads=True):
    """
    Copy files between FileSystems.

    This functions allows you to recursively copy directories of files from
    one file system to another, such as from S3 to your local machine.

    Parameters
    ----------
    source : string
        Source file path or URI to a single file or directory.
        If a directory, files will be copied recursively from this path.
    destination : string
        Destination file path or URI. If `source` is a file, `destination`
        is also interpreted as the destination file (not directory).
        Directories will be created as necessary.
    source_filesystem : FileSystem, optional
        Source filesystem, needs to be specified if `source` is not a URI,
        otherwise inferred.
    destination_filesystem : FileSystem, optional
        Destination filesystem, needs to be specified if `destination` is not
        a URI, otherwise inferred.
    chunk_size : int, default 1MB
        The maximum size of block to read before flushing to the
        destination file. A larger chunk_size will use more memory while
        copying but may help accommodate high latency FileSystems.
    use_threads : bool, default True
        Whether to use multiple threads to accelerate copying.

    Examples
    --------
    Inspect an S3 bucket's files:

    >>> s3, path = fs.FileSystem.from_uri(
    ...            "s3://registry.opendata.aws/roda/ndjson/")
    >>> selector = fs.FileSelector(path)
    >>> s3.get_file_info(selector)
    [<FileInfo for 'registry.opendata.aws/roda/ndjson/index.ndjson':...]

    Copy one file from S3 bucket to a local directory:

    >>> fs.copy_files("s3://registry.opendata.aws/roda/ndjson/index.ndjson",
    ...               "file:///{}/index_copy.ndjson".format(local_path))

    >>> fs.LocalFileSystem().get_file_info(str(local_path)+
    ...                                    '/index_copy.ndjson')
    <FileInfo for '.../index_copy.ndjson': type=FileType.File, size=...>

    Copy file using a FileSystem object:

    >>> fs.copy_files("registry.opendata.aws/roda/ndjson/index.ndjson",
    ...               "file:///{}/index_copy.ndjson".format(local_path),
    ...               source_filesystem=fs.S3FileSystem())
    """
    source_fs, source_path = _resolve_filesystem_and_path(source, source_filesystem)
    destination_fs, destination_path = _resolve_filesystem_and_path(destination, destination_filesystem)
    file_info = source_fs.get_file_info(source_path)
    if file_info.type == FileType.Directory:
        source_sel = FileSelector(source_path, recursive=True)
        _copy_files_selector(source_fs, source_sel, destination_fs, destination_path, chunk_size, use_threads)
    else:
        _copy_files(source_fs, source_path, destination_fs, destination_path, chunk_size, use_threads)