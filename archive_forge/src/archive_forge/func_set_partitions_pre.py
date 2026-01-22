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
def set_partitions_pre(s, divisions, ascending=True, na_position='last'):
    try:
        if ascending:
            partitions = divisions.searchsorted(s, side='right') - 1
        else:
            partitions = len(divisions) - divisions.searchsorted(s, side='right') - 1
    except (TypeError, ValueError):
        partitions = np.empty(len(s), dtype='int32')
        not_null = s.notna()
        divisions_notna = divisions[divisions.notna()]
        if ascending:
            partitions[not_null] = divisions_notna.searchsorted(s[not_null], side='right') - 1
        else:
            partitions[not_null] = len(divisions) - divisions_notna.searchsorted(s[not_null], side='right') - 1
    partitions[(partitions < 0) | (partitions >= len(divisions) - 1)] = len(divisions) - 2 if ascending else 0
    nas = s.isna()
    nas = getattr(nas, 'values', nas)
    partitions[nas] = len(divisions) - 2 if na_position == 'last' else 0
    return partitions