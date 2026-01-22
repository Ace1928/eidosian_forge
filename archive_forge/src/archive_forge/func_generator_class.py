from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
@pytest.fixture(params=[da.random.RandomState, da.random.default_rng])
def generator_class(request):
    return request.param