import ctypes
import re
from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
def from_dataframe_to_pandas(df: ProtocolDataframe, n_chunks: Optional[int]=None):
    """
    Build a ``pandas.DataFrame`` from an object supporting the DataFrame exchange protocol, i.e. `__dataframe__` method.

    Parameters
    ----------
    df : ProtocolDataframe
        Object supporting the exchange protocol, i.e. `__dataframe__` method.
    n_chunks : int, optional
        Number of chunks to split `df`.

    Returns
    -------
    pandas.DataFrame
    """
    if not hasattr(df, '__dataframe__'):
        raise ValueError('`df` does not support __dataframe__')
    df = df.__dataframe__()
    if isinstance(df, dict):
        df = df['dataframe']
    pandas_dfs = []
    for chunk in df.get_chunks(n_chunks):
        pandas_df = protocol_df_chunk_to_pandas(chunk)
        pandas_dfs.append(pandas_df)
    pandas_df = pandas.concat(pandas_dfs, axis=0, ignore_index=True)
    index_obj = df.metadata.get('modin.index', df.metadata.get('pandas.index', None))
    if index_obj is not None:
        pandas_df.index = index_obj
    return pandas_df