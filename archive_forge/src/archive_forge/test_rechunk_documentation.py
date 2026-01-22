from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
Similar to test_rechunk_same_fully_unknown but testing the behavior if
    ``float("nan")`` is used instead of the recommended ``np.nan``
    