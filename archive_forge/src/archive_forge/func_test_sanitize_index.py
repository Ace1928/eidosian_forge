from __future__ import annotations
import itertools
import warnings
import pytest
from tlz import merge
import dask
import dask.array as da
from dask import config
from dask.array.chunk import getitem
from dask.array.slicing import (
from dask.array.utils import assert_eq, same_keys
def test_sanitize_index():
    pd = pytest.importorskip('pandas')
    with pytest.raises(TypeError):
        sanitize_index('Hello!')
    np.testing.assert_equal(sanitize_index(pd.Series([1, 2, 3])), [1, 2, 3])
    np.testing.assert_equal(sanitize_index((1, 2, 3)), [1, 2, 3])