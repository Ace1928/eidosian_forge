import os
from pyarrow.pandas_compat import _pandas_api  # noqa
from pyarrow.lib import (Codec, Table,  # noqa
import pyarrow.lib as ext
from pyarrow import _feather
from pyarrow._feather import FeatherError  # noqa: F401
def write_feather(df, dest, compression=None, compression_level=None, chunksize=None, version=2):
    """
    Write a pandas.DataFrame to Feather format.

    Parameters
    ----------
    df : pandas.DataFrame or pyarrow.Table
        Data to write out as Feather format.
    dest : str
        Local destination path.
    compression : string, default None
        Can be one of {"zstd", "lz4", "uncompressed"}. The default of None uses
        LZ4 for V2 files if it is available, otherwise uncompressed.
    compression_level : int, default None
        Use a compression level particular to the chosen compressor. If None
        use the default compression level
    chunksize : int, default None
        For V2 files, the internal maximum size of Arrow RecordBatch chunks
        when writing the Arrow IPC file format. None means use the default,
        which is currently 64K
    version : int, default 2
        Feather file version. Version 2 is the current. Version 1 is the more
        limited legacy format
    """
    if _pandas_api.have_pandas:
        if _pandas_api.has_sparse and isinstance(df, _pandas_api.pd.SparseDataFrame):
            df = df.to_dense()
    if _pandas_api.is_data_frame(df):
        if version == 1:
            preserve_index = False
        elif version == 2:
            preserve_index = None
        else:
            raise ValueError('Version value should either be 1 or 2')
        table = Table.from_pandas(df, preserve_index=preserve_index)
        if version == 1:
            for i, name in enumerate(table.schema.names):
                col = table[i]
                check_chunked_overflow(name, col)
    else:
        table = df
    if version == 1:
        if len(table.column_names) > len(set(table.column_names)):
            raise ValueError('cannot serialize duplicate column names')
        if compression is not None:
            raise ValueError('Feather V1 files do not support compression option')
        if chunksize is not None:
            raise ValueError('Feather V1 files do not support chunksize option')
    elif compression is None and Codec.is_available('lz4_frame'):
        compression = 'lz4'
    elif compression is not None and compression not in _FEATHER_SUPPORTED_CODECS:
        raise ValueError('compression="{}" not supported, must be one of {}'.format(compression, _FEATHER_SUPPORTED_CODECS))
    try:
        _feather.write_feather(table, dest, compression=compression, compression_level=compression_level, chunksize=chunksize, version=version)
    except Exception:
        if isinstance(dest, str):
            try:
                os.remove(dest)
            except os.error:
                pass
        raise