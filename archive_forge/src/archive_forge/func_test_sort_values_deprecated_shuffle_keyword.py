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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='not deprecated')
def test_sort_values_deprecated_shuffle_keyword(shuffle_method):
    np.random.seed(0)
    df = pd.DataFrame()
    df['a'] = np.ascontiguousarray(np.arange(10)[::-1])
    df['b'] = np.arange(100, 10 + 100)
    ddf = dd.from_pandas(df, npartitions=10)
    with pytest.warns(FutureWarning, match="'shuffle' keyword is deprecated"):
        got = ddf.sort_values(by=['a'], shuffle=shuffle_method)
    expect = df.sort_values(by=['a'])
    dd.assert_eq(got, expect, check_index=False, sort_results=False)