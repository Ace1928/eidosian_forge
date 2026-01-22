from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
def test_raises_bad_kwarg(generator_class):
    with pytest.raises(Exception) as info:
        generator_class().standard_cauchy(size=(10,), dtype='float64')
    assert 'dtype' in str(info.value)