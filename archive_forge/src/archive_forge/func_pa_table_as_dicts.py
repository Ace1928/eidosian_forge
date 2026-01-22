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
def pa_table_as_dicts(df: pa.Table, columns: Optional[List[str]]=None) -> List[Dict[str, Any]]:
    """Convert a pyarrow table to a list of dict

    :param df: pyarrow table
    :param columns: if not None, only these columns will be returned, defaults to None
    :return: a list of dict
    """
    res: List[Dict[str, Any]] = []
    for ck in _pa_table_as_dicts_chunks(df, columns=columns):
        res += ck
    return res