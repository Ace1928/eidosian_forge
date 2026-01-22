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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='auto not supported')
def test_set_index_doesnt_increase_partitions(shuffle_method):
    nparts = 2
    nbytes = 1000000.0
    n = int(nbytes / (nparts * 8))
    ddf = dd.DataFrame({('x', i): (make_part, n) for i in range(nparts)}, 'x', make_part(1), [None] * (nparts + 1))
    ddf2 = ddf.set_index('x', shuffle_method=shuffle_method, partition_size=nbytes)
    assert ddf2.npartitions <= ddf.npartitions