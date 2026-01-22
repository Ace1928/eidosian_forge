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
def test_shuffle_from_one_partition_to_one_other(shuffle_method):
    df = pd.DataFrame({'x': [1, 2, 3]})
    a = dd.from_pandas(df, 1)
    for i in [1, 2]:
        b = shuffle(a, 'x', npartitions=i, shuffle_method=shuffle_method)
        assert len(a.compute(scheduler='sync')) == len(b.compute(scheduler='sync'))