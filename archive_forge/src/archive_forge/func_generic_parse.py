import contextlib
import json
import os
import warnings
from io import BytesIO, IOBase, TextIOWrapper
from typing import Any, NamedTuple
import fsspec
import numpy as np
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.concat import union_categoricals
from pandas.io.common import infer_compression
from pandas.util._decorators import doc
from modin.config import MinPartitionSize
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.db_conn import ModinDatabaseConnection
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import ModinAssumptionError
@staticmethod
@doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
def generic_parse(fname, **kwargs):
    warnings.filterwarnings('ignore')
    num_splits = kwargs.pop('num_splits', None)
    start = kwargs.pop('start', None)
    end = kwargs.pop('end', None)
    header_size = kwargs.pop('header_size', 0)
    common_dtypes = kwargs.pop('common_dtypes', None)
    encoding = kwargs.get('encoding', None)
    callback = kwargs.pop('callback')
    if start is None or end is None:
        return callback(fname, **kwargs)
    with OpenFile(fname, 'rb', kwargs.pop('compression', 'infer'), **kwargs.pop('storage_options', None) or {}) as bio:
        header = b''
        if encoding and ('utf' in encoding and '8' not in encoding or encoding == 'unicode_escape' or encoding.replace('-', '_') == 'utf_8_sig'):
            fio = TextIOWrapper(bio, encoding=encoding, newline='')
            if header_size == 0:
                header = fio.readline().encode(encoding)
                kwargs['skiprows'] = 1
            for _ in range(header_size):
                header += fio.readline().encode(encoding)
        elif encoding is not None:
            if header_size == 0:
                header = bio.readline()
                kwargs['skiprows'] = 1
            for _ in range(header_size):
                header += bio.readline()
        else:
            for _ in range(header_size):
                header += bio.readline()
        bio.seek(start)
        to_read = header + bio.read(end - start)
    if 'memory_map' in kwargs:
        kwargs = kwargs.copy()
        del kwargs['memory_map']
    if common_dtypes is not None:
        kwargs['dtype'] = common_dtypes
    pandas_df = callback(BytesIO(to_read), **kwargs)
    index = pandas_df.index if not isinstance(pandas_df.index, pandas.RangeIndex) else len(pandas_df)
    return _split_result_for_readers(1, num_splits, pandas_df) + [index, pandas_df.dtypes]