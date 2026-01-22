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
@pytest.mark.parametrize('ascending', [True, False])
@pytest.mark.parametrize('by', ['a', 'b', ['a', 'b']])
@pytest.mark.parametrize('nelem', [10, 500])
def test_sort_values(nelem, by, ascending):
    np.random.seed(0)
    df = pd.DataFrame()
    df['a'] = np.ascontiguousarray(np.arange(nelem)[::-1])
    df['b'] = np.arange(100, nelem + 100)
    ddf = dd.from_pandas(df, npartitions=10)
    with dask.config.set(scheduler='single-threaded'):
        got = ddf.sort_values(by=by, ascending=ascending)
    expect = df.sort_values(by=by, ascending=ascending)
    dd.assert_eq(got, expect, check_index=False, sort_results=False)