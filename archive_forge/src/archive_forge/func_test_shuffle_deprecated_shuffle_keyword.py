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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='no deprecation necessary')
def test_shuffle_deprecated_shuffle_keyword(shuffle_method):
    from dask.dataframe.tests.test_multi import list_eq
    with pytest.warns(FutureWarning, match="'shuffle' keyword is deprecated"):
        result = dd.shuffle.shuffle(d, d.b, shuffle=shuffle_method)
    list_eq(result, d)
    with pytest.warns(FutureWarning, match="'shuffle' keyword is deprecated"):
        result = d.shuffle(d.b, shuffle=shuffle_method)
    list_eq(result, d)