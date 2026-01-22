from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
def test_unified_dim_sizes() -> None:
    assert unified_dim_sizes([xr.Variable((), 0)]) == {}
    assert unified_dim_sizes([xr.Variable('x', [1]), xr.Variable('x', [1])]) == {'x': 1}
    assert unified_dim_sizes([xr.Variable('x', [1]), xr.Variable('y', [1, 2])]) == {'x': 1, 'y': 2}
    assert unified_dim_sizes([xr.Variable(('x', 'z'), [[1]]), xr.Variable(('y', 'z'), [[1, 2], [3, 4]])], exclude_dims={'z'}) == {'x': 1, 'y': 2}
    with pytest.raises(ValueError):
        unified_dim_sizes([xr.Variable(('x', 'x'), [[1]])])
    with pytest.raises(ValueError):
        unified_dim_sizes([xr.Variable('x', [1]), xr.Variable('x', [1, 2])])