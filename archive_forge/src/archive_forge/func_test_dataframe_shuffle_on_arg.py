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
@pytest.mark.parametrize('ignore_index', [None, True, False])
@pytest.mark.parametrize('on', ['id', 'name', ['id', 'name'], pd.Series(['id', 'name'])])
@pytest.mark.parametrize('max_branch', [None, 4])
def test_dataframe_shuffle_on_arg(on, ignore_index, max_branch, shuffle_method):
    df_in = dask.datasets.timeseries('2000', '2001', types={'value': float, 'name': str, 'id': int}, freq='2h', partition_freq=f'1{ME}', seed=1)
    if isinstance(on, str):
        ext_on = df_in[[on]].copy()
    else:
        ext_on = df_in[on].copy()
    df_out_1 = df_in.shuffle(on, shuffle_method=shuffle_method, ignore_index=ignore_index, max_branch=max_branch)
    df_out_2 = df_in.shuffle(ext_on, shuffle_method=shuffle_method, ignore_index=ignore_index)
    assert_eq(df_out_1, df_out_2, check_index=not ignore_index)
    if ignore_index and shuffle_method == 'tasks':
        assert df_out_1.index.dtype != df_in.index.dtype
    else:
        assert df_out_1.index.dtype == df_in.index.dtype