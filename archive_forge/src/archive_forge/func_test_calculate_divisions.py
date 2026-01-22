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
@pytest.mark.parametrize('pdf,expected', [(pd.DataFrame({'x': list('aabbcc'), 'y': list('xyyyzz')}), (['a', 'b', 'c', 'c'], ['a', 'b', 'c', 'c'], ['a', 'b', 'c', 'c'], False)), (pd.DataFrame({'x': [1, 0, 1, 3, 4, 5, 7, 8, 1, 2, 3], 'y': [21, 9, 7, 8, 3, 5, 4, 5, 6, 3, 10]}), ([0, 1, 2, 4, 8], [0, 3, 1, 2], [1, 5, 8, 3], False)), (pd.DataFrame({'x': [5, 6, 7, 10, None, 10, 2, None, 8, 4, None]}), ([2.0, 4.0, 5.666666666666667, 8.0, 10.0], [5.0, 10.0, 2.0, 4.0], [7.0, 10.0, 8.0, 4.0], False))])
def test_calculate_divisions(pdf, expected):
    ddf = dd.from_pandas(pdf, npartitions=4)
    divisions, mins, maxes, presorted = _calculate_divisions(ddf, ddf['x'], False, 4)
    assert divisions == expected[0]
    assert mins == expected[1]
    assert maxes == expected[2]
    assert presorted == expected[3]