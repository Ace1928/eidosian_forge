import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from fsspec import AbstractFileSystem
from triad import Schema, assert_or_throw
from triad.collections.schema import SchemaError
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.io import url_to_fs
from triad.utils.pyarrow import pa_batch_to_dicts
from .api import as_fugue_df, get_column_names, normalize_column_names, rename
from .dataframe import DataFrame, LocalBoundedDataFrame
def deserialize_df(data: Optional[bytes], fs: Optional[AbstractFileSystem]=None) -> Optional[LocalBoundedDataFrame]:
    """Deserialize json string to
    :class:`~fugue.dataframe.dataframe.LocalBoundedDataFrame`

    :param json_str: json string containing the base64 data or a file path
    :param fs: the file system to use, defaults to None
    :raises ValueError: if the json string is invalid, not generated from
      :func:`~.serialize_df`
    :return: :class:`~fugue.dataframe.dataframe.LocalBoundedDataFrame` if ``json_str``
      contains a dataframe or None if its valid but contains no data
    """
    if data is None:
        return None
    obj = pickle.loads(data)
    if isinstance(obj, LocalBoundedDataFrame):
        return obj
    elif isinstance(obj, str):
        fs, path = url_to_fs(obj)
        with fs.open(path, 'rb') as f:
            return pickle.load(f)
    raise ValueError('data is invalid')