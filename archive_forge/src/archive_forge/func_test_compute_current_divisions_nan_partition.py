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
def test_compute_current_divisions_nan_partition():
    a = d[d.a > 3].sort_values('a')
    divisions = a.compute_current_divisions('a')
    assert divisions == (4, 5, 8, 9)
    if DASK_EXPR_ENABLED:
        pass
    else:
        a.divisions = divisions
    assert_eq(a, a, check_divisions=False)
    a = d[d.a > 1].sort_values('a')
    divisions = a.compute_current_divisions('a')
    assert divisions == (2, 4, 7, 9)
    if DASK_EXPR_ENABLED:
        pass
    else:
        a.divisions = divisions
    assert_eq(a, a, check_divisions=False)