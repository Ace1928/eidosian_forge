from __future__ import annotations
import os
from collections.abc import Mapping
from io import BytesIO
from warnings import catch_warnings, simplefilter, warn
import numpy as np
import pandas as pd
from fsspec.compression import compr
from fsspec.core import get_fs_token_paths
from fsspec.core import open as open_file
from fsspec.core import open_files
from fsspec.utils import infer_compression
from pandas.api.types import (
from dask.base import tokenize
from dask.bytes import read_bytes
from dask.core import flatten
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.io.io import from_map
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.dataframe.utils import clear_known_categories
from dask.delayed import delayed
from dask.utils import asciitable, parse_bytes
from the start of the file (or of the first file if it's a glob). Usually this
from dask.dataframe.core import _Frame
def read_pandas(reader, urlpath, blocksize='default', lineterminator=None, compression='infer', sample=256000, sample_rows=10, enforce=False, assume_missing=False, storage_options=None, include_path_column=False, **kwargs):
    reader_name = reader.__name__
    if lineterminator is not None and len(lineterminator) == 1:
        kwargs['lineterminator'] = lineterminator
    else:
        lineterminator = '\n'
    if 'encoding' in kwargs:
        b_lineterminator = lineterminator.encode(kwargs['encoding'])
        empty_blob = ''.encode(kwargs['encoding'])
        if empty_blob:
            b_lineterminator = b_lineterminator[len(empty_blob):]
    else:
        b_lineterminator = lineterminator.encode()
    if include_path_column and isinstance(include_path_column, bool):
        include_path_column = 'path'
    if 'index' in kwargs or ('index_col' in kwargs and kwargs.get('index_col') is not False):
        raise ValueError("Keywords 'index' and 'index_col' not supported, except for 'index_col=False'. Use dd.{reader_name}(...).set_index('my-index') instead")
    for kw in ['iterator', 'chunksize']:
        if kw in kwargs:
            raise ValueError(f'{kw} not supported for dd.{reader_name}')
    if kwargs.get('nrows', None):
        raise ValueError("The 'nrows' keyword is not supported by `dd.{0}`. To achieve the same behavior, it's recommended to use `dd.{0}(...).head(n=nrows)`".format(reader_name))
    if isinstance(kwargs.get('skiprows'), int):
        lastskiprow = firstrow = kwargs.get('skiprows')
    elif kwargs.get('skiprows') is None:
        lastskiprow = firstrow = 0
    else:
        skiprows = set(kwargs.get('skiprows'))
        lastskiprow = max(skiprows)
        firstrow = min(set(range(len(skiprows) + 1)) - set(skiprows))
    if isinstance(kwargs.get('header'), list):
        raise TypeError(f'List of header rows not supported for dd.{reader_name}')
    if isinstance(kwargs.get('converters'), dict) and include_path_column:
        path_converter = kwargs.get('converters').get(include_path_column, None)
    else:
        path_converter = None
    if compression == 'infer':
        paths = get_fs_token_paths(urlpath, mode='rb', storage_options=storage_options)[2]
        if len(paths) == 0:
            raise OSError(f'{urlpath} resolved to no files')
        compression = infer_compression(paths[0])
    if blocksize == 'default':
        blocksize = AUTO_BLOCKSIZE
    if isinstance(blocksize, str):
        blocksize = parse_bytes(blocksize)
    if blocksize and compression:
        warn('Warning %s compression does not support breaking apart files\nPlease ensure that each individual file can fit in memory and\nuse the keyword ``blocksize=None to remove this message``\nSetting ``blocksize=None``' % compression)
        blocksize = None
    if compression not in compr:
        raise NotImplementedError('Compression format %s not installed' % compression)
    if blocksize and sample and (blocksize < sample) and (lastskiprow != 0):
        warn('Unexpected behavior can result from passing skiprows when\nblocksize is smaller than sample size.\nSetting ``sample=blocksize``')
        sample = blocksize
    b_out = read_bytes(urlpath, delimiter=b_lineterminator, blocksize=blocksize, sample=sample, compression=compression, include_path=include_path_column, **storage_options or {})
    if include_path_column:
        b_sample, values, paths = b_out
        path = (include_path_column, path_converter)
    else:
        b_sample, values = b_out
        path = None
    if not isinstance(values[0], (tuple, list)):
        values = [values]
    if b_sample is False and len(values[0]):
        b_sample = values[0][0].compute()
    names = kwargs.get('names', None)
    header = kwargs.get('header', 'infer' if names is None else None)
    need = 1 if header is None else 2
    if isinstance(header, int):
        firstrow += header
    if kwargs.get('comment'):
        parts = []
        for part in b_sample.split(b_lineterminator):
            split_comment = part.decode().split(kwargs.get('comment'))
            if len(split_comment) > 1:
                if len(split_comment[0]) > 0:
                    parts.append(split_comment[0].strip().encode())
            else:
                parts.append(part)
            if len(parts) > need:
                break
    else:
        parts = b_sample.split(b_lineterminator, max(lastskiprow + need, firstrow + need))
    nparts = 0 if not parts else len(parts) - int(not parts[-1])
    if sample is not False and nparts < lastskiprow + need and (len(b_sample) >= sample):
        raise ValueError('Sample is not large enough to include at least one row of data. Please increase the number of bytes in `sample` in the call to `read_csv`/`read_table`')
    header = b'' if header is None else parts[firstrow] + b_lineterminator
    head_kwargs = kwargs.copy()
    head_kwargs.pop('skipfooter', None)
    if head_kwargs.get('engine') == 'pyarrow':
        head_kwargs['engine'] = 'c'
    try:
        head = reader(BytesIO(b_sample), nrows=sample_rows, **head_kwargs)
    except pd.errors.ParserError as e:
        if 'EOF' in str(e):
            raise ValueError('EOF encountered while reading header. \nPass argument `sample_rows` and make sure the value of `sample` is large enough to accommodate that many rows of data') from e
        raise
    if include_path_column and include_path_column in head.columns:
        raise ValueError('Files already contain the column name: %s, so the path column cannot use this name. Please set `include_path_column` to a unique name.' % include_path_column)
    specified_dtypes = kwargs.get('dtype', {})
    if specified_dtypes is None:
        specified_dtypes = {}
    if assume_missing and isinstance(specified_dtypes, dict):
        for c in head.columns:
            if is_integer_dtype(head[c].dtype) and c not in specified_dtypes:
                head[c] = head[c].astype(float)
    values = [[list(dsk.dask.values()) for dsk in block] for block in values]
    return text_blocks_to_pandas(reader, values, header, head, kwargs, enforce=enforce, specified_dtypes=specified_dtypes, path=path, blocksize=blocksize, urlpath=urlpath)