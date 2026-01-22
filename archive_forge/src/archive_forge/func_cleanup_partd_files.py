from __future__ import annotations
import contextlib
import logging
import math
import shutil
import tempfile
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal
import numpy as np
import pandas as pd
import tlz as toolz
from pandas.api.types import is_numeric_dtype
from dask import config
from dask.base import compute, compute_as_if_collection, is_dask_collection, tokenize
from dask.dataframe import methods
from dask.dataframe._compat import PANDAS_GE_300
from dask.dataframe.core import (
from dask.dataframe.dispatch import (
from dask.dataframe.utils import UNKNOWN_CATEGORIES
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ShuffleLayer, SimpleShuffleLayer
from dask.sizeof import sizeof
from dask.utils import M, digit, get_default_shuffle_method
def cleanup_partd_files(p, keys):
    """
    Cleanup the files in a partd.File dataset.

    Parameters
    ----------
    p : partd.Interface
        File or Encode wrapping a file should be OK.
    keys: List
        Just for scheduling purposes, not actually used.
    """
    import partd
    if isinstance(p, partd.Encode):
        maybe_file = p.partd
    else:
        maybe_file = None
    if isinstance(maybe_file, partd.File):
        path = maybe_file.path
    else:
        path = None
    if path:
        shutil.rmtree(path, ignore_errors=True)