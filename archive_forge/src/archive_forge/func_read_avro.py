from __future__ import annotations
import io
import uuid
from fsspec.core import OpenFile, get_fs_token_paths, open_files
from fsspec.utils import read_block
from fsspec.utils import tokenize as fs_tokenize
from dask.highlevelgraph import HighLevelGraph
def read_avro(urlpath, blocksize=100000000, storage_options=None, compression=None):
    """Read set of avro files

    Use this with arbitrary nested avro schemas. Please refer to the
    fastavro documentation for its capabilities:
    https://github.com/fastavro/fastavro

    Parameters
    ----------
    urlpath: string or list
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``), or globstring pointing to data.
    blocksize: int or None
        Size of chunks in bytes. If None, there will be no chunking and each
        file will become one partition.
    storage_options: dict or None
        passed to backend file-system
    compression: str or None
        Compression format of the targe(s), like 'gzip'. Should only be used
        with blocksize=None.
    """
    from dask import compute, delayed
    from dask.bag import from_delayed
    from dask.utils import import_required
    import_required('fastavro', 'fastavro is a required dependency for using bag.read_avro().')
    storage_options = storage_options or {}
    if blocksize is not None:
        fs, fs_token, paths = get_fs_token_paths(urlpath, mode='rb', storage_options=storage_options)
        dhead = delayed(open_head)
        out = compute(*[dhead(fs, path, compression) for path in paths])
        heads, sizes = zip(*out)
        dread = delayed(read_chunk)
        offsets = []
        lengths = []
        for size in sizes:
            off = list(range(0, size, blocksize))
            length = [blocksize] * len(off)
            offsets.append(off)
            lengths.append(length)
        out = []
        for path, offset, length, head in zip(paths, offsets, lengths, heads):
            delimiter = head['sync']
            f = OpenFile(fs, path, compression=compression)
            token = fs_tokenize(fs_token, delimiter, path, fs.ukey(path), compression, offset)
            keys = [f'read-avro-{o}-{token}' for o in offset]
            values = [dread(f, o, l, head, dask_key_name=key) for o, key, l in zip(offset, keys, length)]
            out.extend(values)
        return from_delayed(out)
    else:
        files = open_files(urlpath, compression=compression, **storage_options)
        dread = delayed(read_file)
        chunks = [dread(fo) for fo in files]
        return from_delayed(chunks)