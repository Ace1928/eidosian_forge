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
def test_set_index_with_dask_dt_index():
    values = {'x': [1, 2, 3, 4] * 3, 'y': [10, 20, 30] * 4, 'name': ['Alice', 'Bob'] * 6}
    date_index = pd.date_range(start='2022-02-22', freq='16h', periods=12) - pd.Timedelta(seconds=30)
    df = pd.DataFrame(values, index=date_index)
    ddf = dd.from_pandas(df, npartitions=3)
    day_index = ddf.index.dt.floor('D')
    day_df = ddf.set_index(day_index)
    expected = dd.from_pandas(pd.DataFrame(values, index=date_index.floor('D')), npartitions=3)
    assert_eq(day_df, expected)
    one_day = pd.Timedelta(days=1)
    next_day_df = ddf.set_index(ddf.index + one_day)
    expected = dd.from_pandas(pd.DataFrame(values, index=date_index + one_day), npartitions=3)
    assert_eq(next_day_df, expected)
    no_dates = dd.from_pandas(pd.DataFrame(values), npartitions=3)
    range_df = ddf.set_index(no_dates.index)
    expected = dd.from_pandas(pd.DataFrame(values), npartitions=3)
    assert_eq(range_df, expected)