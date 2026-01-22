from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
def test_can_make_really_big_random_array(generator_class):
    generator_class().normal(10, 1, (1000000, 1000000), chunks=(100000, 100000))