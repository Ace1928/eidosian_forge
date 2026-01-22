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
def test_disk_shuffle_check_actual_compression():

    def generate_raw_partd_file(compression):
        df1 = pd.DataFrame({'a': list(range(10000))})
        df1['b'] = (df1['a'] * 123).astype(str)
        with dask.config.set({'dataframe.shuffle.compression': compression}):
            p1 = maybe_buffered_partd(buffer=False, tempdir=None)()
            p1.append({'x': df1})
            filename = p1.partd.partd.filename('x') if compression else p1.partd.filename('x')
            with open(filename, 'rb') as f:
                return f.read()
    uncompressed_data = generate_raw_partd_file(compression=None)
    compressed_data = generate_raw_partd_file(compression='BZ2')
    assert len(uncompressed_data) > len(compressed_data)