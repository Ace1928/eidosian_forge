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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='we test this over in dask-expr')
def test_set_index_does_not_repeat_work_due_to_optimizations():
    count = itertools.count()

    def increment():
        next(count)

    def make_part(dummy, n):
        return pd.DataFrame({'x': np.random.random(n), 'y': np.random.random(n)})
    nbytes = 1000000.0
    nparts = 50
    n = int(nbytes / (nparts * 8))
    dsk = {('inc', i): (increment,) for i in range(nparts)}
    dsk.update({('x', i): (make_part, ('inc', i), n) for i in range(nparts)})
    ddf = dd.DataFrame(dsk, 'x', make_part(None, 1), [None] * (nparts + 1))
    ddf.set_index('x')
    ntimes = next(count)
    assert ntimes == nparts