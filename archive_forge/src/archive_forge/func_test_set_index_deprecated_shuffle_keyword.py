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
def test_set_index_deprecated_shuffle_keyword(shuffle_method):
    df = pd.DataFrame({'x': [4, 1, 1, 3, 3], 'y': [1.0, 1, 1, 1, 2]})
    ddf = dd.from_pandas(df, 2)
    expected = df.set_index('x')
    with pytest.warns(FutureWarning, match="'shuffle' keyword is deprecated"):
        result = ddf.set_index('x', shuffle=shuffle_method)
    assert_eq(result, expected)
    with pytest.warns(FutureWarning, match="'shuffle' keyword is deprecated"):
        result = dd.shuffle.set_index(ddf, 'x', shuffle=shuffle_method)
    assert_eq(result, expected)