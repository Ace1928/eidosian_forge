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
@pytest.mark.skipif(not PANDAS_GE_150, reason='Only test `string[pyarrow]` on recent versions of pandas')
@pytest.mark.parametrize('string_dtype', ['string[python]', 'string[pyarrow]', 'object'])
def test_set_index_string(shuffle_method, string_dtype):
    if string_dtype == 'string[pyarrow]':
        pytest.importorskip('pyarrow')
    names = ['alice', 'bob', 'ricky']
    df = pd.DataFrame({'x': np.random.random(100), 'y': np.random.choice(names, 100)}, index=np.random.random(100))
    df = df.astype({'y': string_dtype})
    ddf = dd.from_pandas(df, npartitions=10)
    assert_eq(df.set_index('y'), ddf.set_index('y', shuffle_method=shuffle_method))