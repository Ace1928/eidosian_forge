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
def text_blocks_to_pandas(reader, block_lists, header, head, kwargs, enforce=False, specified_dtypes=None, path=None, blocksize=None, urlpath=None):
    """Convert blocks of bytes to a dask.dataframe

    This accepts a list of lists of values of bytes where each list corresponds
    to one file, and the value of bytes concatenate to comprise the entire
    file, in order.

    Parameters
    ----------
    reader : callable
        ``pd.read_csv`` or ``pd.read_table``.
    block_lists : list of lists of delayed values of bytes
        The lists of bytestrings where each list corresponds to one logical file
    header : bytestring
        The header, found at the front of the first file, to be prepended to
        all blocks
    head : pd.DataFrame
        An example Pandas DataFrame to be used for metadata.
    kwargs : dict
        Keyword arguments to pass down to ``reader``
    path : tuple, optional
        A tuple containing column name for path and the path_converter if provided

    Returns
    -------
    A dask.dataframe
    """
    dtypes = head.dtypes.to_dict()
    categoricals = head.select_dtypes(include=['category']).columns
    if isinstance(specified_dtypes, Mapping):
        known_categoricals = [k for k in categoricals if isinstance(specified_dtypes.get(k), CategoricalDtype) and specified_dtypes.get(k).categories is not None]
        unknown_categoricals = categoricals.difference(known_categoricals)
    else:
        unknown_categoricals = categoricals
    for k in unknown_categoricals:
        dtypes[k] = 'category'
    columns = list(head.columns)
    blocks = tuple(flatten(block_lists))
    is_first = tuple(block_mask(block_lists))
    is_last = tuple(block_mask_last(block_lists))
    if path:
        colname, path_converter = path
        paths = [b[1].path for b in blocks]
        if path_converter:
            paths = [path_converter(p) for p in paths]
        head = head.assign(**{colname: pd.Categorical.from_codes(np.zeros(len(head), dtype=int), set(paths))})
        path = (colname, paths)
    if len(unknown_categoricals):
        head = clear_known_categories(head, cols=unknown_categoricals)
    parts = []
    colname, paths = path or (None, None)
    for i in range(len(blocks)):
        parts.append([blocks[i], paths[i] if paths else None, is_first[i], is_last[i]])
    return from_map(CSVFunctionWrapper(columns, None, colname, head, header, reader, dtypes, enforce, kwargs), parts, meta=head, label='read-csv', token=tokenize(reader, urlpath, columns, enforce, head, blocksize), enforce_metadata=False, produces_tasks=True)