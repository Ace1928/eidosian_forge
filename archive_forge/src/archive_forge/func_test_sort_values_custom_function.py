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
@pytest.mark.parametrize('by', [['a', 'b'], ['b', 'a']])
@pytest.mark.parametrize('nparts', [1, 10])
def test_sort_values_custom_function(by, nparts):
    df = pd.DataFrame({'a': [1, 2, 3] * 20, 'b': [4, 5, 6, 7] * 15})
    ddf = dd.from_pandas(df, npartitions=nparts)

    def f(partition, by_columns, ascending, na_position, **kwargs):
        return partition.sort_values(by_columns, ascending=ascending, na_position=na_position)
    with dask.config.set(scheduler='single-threaded'):
        got = ddf.sort_values(by=by[0], sort_function=f, sort_function_kwargs={'by_columns': by})
    expect = df.sort_values(by=by)
    dd.assert_eq(got, expect, check_index=False)