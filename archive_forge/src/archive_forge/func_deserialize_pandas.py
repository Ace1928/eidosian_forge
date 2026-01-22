import os
import pyarrow as pa
from pyarrow.lib import (IpcReadOptions, IpcWriteOptions, ReadStats, WriteStats,  # noqa
import pyarrow.lib as lib
def deserialize_pandas(buf, *, use_threads=True):
    """Deserialize a buffer protocol compatible object into a pandas DataFrame.

    Parameters
    ----------
    buf : buffer
        An object compatible with the buffer protocol.
    use_threads : bool, default True
        Whether to parallelize the conversion using multiple threads.

    Returns
    -------
    df : pandas.DataFrame
        The buffer deserialized as pandas DataFrame
    """
    buffer_reader = pa.BufferReader(buf)
    with pa.RecordBatchStreamReader(buffer_reader) as reader:
        table = reader.read_all()
    return table.to_pandas(use_threads=use_threads)