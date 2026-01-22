from __future__ import annotations
import contextlib
import itertools
import multiprocessing as mp
import os
import pickle
import random
import string
import tempfile
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from datetime import date, time
from decimal import Decimal
from functools import partial
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.shuffle import (
from dask.dataframe.utils import assert_eq, make_meta
from dask.optimization import cull
def test_set_index_self_index(shuffle_method):
    df = pd.DataFrame({'x': np.random.random(100), 'y': np.random.random(100) // 0.2}, index=np.random.random(100))
    a = dd.from_pandas(df, npartitions=4)
    if DASK_EXPR_ENABLED:
        ctx = contextlib.nullcontext()
    else:
        ctx = pytest.warns(UserWarning, match='this is a no-op')
    with ctx:
        b = a.set_index(a.index, shuffle_method=shuffle_method)
    assert a is b
    assert_eq(b, df.set_index(df.index))