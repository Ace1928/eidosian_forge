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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason="constructor doesn't work")
def test_set_index_divisions_sorted():
    p1 = pd.DataFrame({'x': [10, 11, 12], 'y': ['a', 'a', 'a']})
    p2 = pd.DataFrame({'x': [13, 14, 15], 'y': ['b', 'b', 'c']})
    p3 = pd.DataFrame({'x': [16, 17, 18], 'y': ['d', 'e', 'e']})
    ddf = dd.DataFrame({('x', 0): p1, ('x', 1): p2, ('x', 2): p3}, 'x', p1, [None, None, None, None])
    df = ddf.compute()

    def throw(*args, **kwargs):
        raise Exception("Shouldn't have computed")
    with dask.config.set(scheduler=throw):
        res = ddf.set_index('x', divisions=[10, 13, 16, 18], sorted=True)
    assert_eq(res, df.set_index('x'))
    with dask.config.set(scheduler=throw):
        res = ddf.set_index('y', divisions=['a', 'b', 'd', 'e'], sorted=True)
    assert_eq(res, df.set_index('y'))
    with pytest.raises(ValueError):
        ddf.set_index('y', divisions=['a', 'b', 'c', 'd', 'e'], sorted=True)
    with pytest.raises(ValueError):
        ddf.set_index('y', divisions=['a', 'b', 'd', 'c'], sorted=True)