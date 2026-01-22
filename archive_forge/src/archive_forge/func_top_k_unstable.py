from pyarrow._compute import (  # noqa
from collections import namedtuple
import inspect
from textwrap import dedent
import warnings
import pyarrow as pa
from pyarrow import _compute_docstrings
from pyarrow.vendored import docscrape
def top_k_unstable(values, k, sort_keys=None, *, memory_pool=None):
    """
    Select the indices of the top-k ordered elements from array- or table-like
    data.

    This is a specialization for :func:`select_k_unstable`. Output is not
    guaranteed to be stable.

    Parameters
    ----------
    values : Array, ChunkedArray, RecordBatch, or Table
        Data to sort and get top indices from.
    k : int
        The number of `k` elements to keep.
    sort_keys : List-like
        Column key names to order by when input is table-like data.
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    result : Array
        Indices of the top-k ordered elements

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>> arr = pa.array(["a", "b", "c", None, "e", "f"])
    >>> pc.top_k_unstable(arr, k=3)
    <pyarrow.lib.UInt64Array object at ...>
    [
      5,
      4,
      2
    ]
    """
    if sort_keys is None:
        sort_keys = []
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        sort_keys.append(('dummy', 'descending'))
    else:
        sort_keys = map(lambda key_name: (key_name, 'descending'), sort_keys)
    options = SelectKOptions(k, sort_keys)
    return call_function('select_k_unstable', [values], options, memory_pool)