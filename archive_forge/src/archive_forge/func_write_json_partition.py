from __future__ import annotations
import io
import os
from functools import partial
from itertools import zip_longest
import pandas as pd
from fsspec.core import open_files
from dask.base import compute as dask_compute
from dask.bytes import read_bytes
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_VERSION
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.io.io import from_delayed
from dask.dataframe.utils import insert_meta_param_description, make_meta
from dask.delayed import delayed
def write_json_partition(df, openfile, kwargs):
    with openfile as f:
        df.to_json(f, **kwargs)
    return os.path.normpath(openfile.path)