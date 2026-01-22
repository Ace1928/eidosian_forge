from __future__ import annotations
import pytest
import numpy as np
import pytest
from tlz import concat
import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.array.numpy_compat import AxisError
from dask.array.utils import assert_eq, same_keys
def test_string_auto_chunk():
    with pytest.raises(ValueError):
        da.full((10000, 10000), 'auto_chunk', chunks='auto')