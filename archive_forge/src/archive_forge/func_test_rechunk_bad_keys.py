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
def test_rechunk_bad_keys():
    x = da.zeros((2, 3, 4), chunks=1)
    assert x.rechunk({-1: 4}).chunks == ((1, 1), (1, 1, 1), (4,))
    assert x.rechunk({-x.ndim: 2}).chunks == ((2,), (1, 1, 1), (1, 1, 1, 1))
    with pytest.raises(TypeError) as info:
        x.rechunk({'blah': 4})
    assert 'blah' in str(info.value)
    with pytest.raises(ValueError) as info:
        x.rechunk({100: 4})
    assert '100' in str(info.value)
    with pytest.raises(ValueError) as info:
        x.rechunk({-100: 4})
    assert '-100' in str(info.value)