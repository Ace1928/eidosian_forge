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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='not valid for dask-expr')
def test_set_index_partitions_meta_dtype():
    ddf = dd.from_pandas(pd.DataFrame({'a': np.random.randint(0, 10, 100)}, index=np.random.random(100)), npartitions=10)
    ddf = ddf.set_index('a', shuffle_method='tasks')
    dsk = ddf.__dask_graph__()
    for layer in dsk.layers.values():
        if isinstance(layer, dd.shuffle.SimpleShuffleLayer):
            assert layer.meta_input['_partitions'].dtype == np.int64