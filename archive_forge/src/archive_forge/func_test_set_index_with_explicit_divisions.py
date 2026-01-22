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
def test_set_index_with_explicit_divisions():
    df = pd.DataFrame({'x': [4, 1, 2, 5]}, index=[10, 20, 30, 40])
    ddf = dd.from_pandas(df, npartitions=2)

    def throw(*args, **kwargs):
        raise Exception()
    with dask.config.set(scheduler=throw):
        ddf2 = ddf.set_index('x', divisions=[1, 3, 5])
    assert ddf2.divisions == (1, 3, 5)
    df2 = df.set_index('x')
    assert_eq(ddf2, df2)
    with pytest.raises(ValueError):
        ddf.set_index('x', divisions=[3, 1, 5])