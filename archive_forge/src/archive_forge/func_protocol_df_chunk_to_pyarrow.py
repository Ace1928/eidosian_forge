from __future__ import annotations
from typing import (
from pyarrow.interchange.column import (
import pyarrow as pa
import re
import pyarrow.compute as pc
from pyarrow.interchange.column import Dtype
def protocol_df_chunk_to_pyarrow(df: DataFrameObject, allow_copy: bool=True) -> pa.RecordBatch:
    """
    Convert interchange protocol chunk to ``pa.RecordBatch``.

    Parameters
    ----------
    df : DataFrameObject
        Object supporting the interchange protocol, i.e. `__dataframe__`
        method.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.RecordBatch
    """
    columns: dict[str, pa.Array] = {}
    for name in df.column_names():
        if not isinstance(name, str):
            raise ValueError(f'Column {name} is not a string')
        if name in columns:
            raise ValueError(f'Column {name} is not unique')
        col = df.get_column_by_name(name)
        dtype = col.dtype[0]
        if dtype in (DtypeKind.INT, DtypeKind.UINT, DtypeKind.FLOAT, DtypeKind.STRING, DtypeKind.DATETIME):
            columns[name] = column_to_array(col, allow_copy)
        elif dtype == DtypeKind.BOOL:
            columns[name] = bool_column_to_array(col, allow_copy)
        elif dtype == DtypeKind.CATEGORICAL:
            columns[name] = categorical_column_to_dictionary(col, allow_copy)
        else:
            raise NotImplementedError(f'Data type {dtype} not handled yet')
    return pa.RecordBatch.from_pydict(columns)