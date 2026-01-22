from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_stack_unstack(self) -> None:
    orig = DataArray([[0, 1], [2, 3]], dims=['x', 'y'], attrs={'foo': 2})
    assert_identical(orig, orig.unstack())
    a = orig[:0, :1].stack(dim=('x', 'y')).indexes['dim']
    b = pd.MultiIndex(levels=[pd.Index([], np.int64), pd.Index([0], np.int64)], codes=[[], []], names=['x', 'y'])
    pd.testing.assert_index_equal(a, b)
    actual = orig.stack(z=['x', 'y']).unstack('z').drop_vars(['x', 'y'])
    assert_identical(orig, actual)
    actual = orig.stack(z=[...]).unstack('z').drop_vars(['x', 'y'])
    assert_identical(orig, actual)
    dims = ['a', 'b', 'c', 'd', 'e']
    coords = {'a': [0], 'b': [1, 2], 'c': [3, 4, 5], 'd': [6, 7], 'e': [8]}
    orig = xr.DataArray(np.random.rand(1, 2, 3, 2, 1), coords=coords, dims=dims)
    stacked = orig.stack(ab=['a', 'b'], cd=['c', 'd'])
    unstacked = stacked.unstack(['ab', 'cd'])
    assert_identical(orig, unstacked.transpose(*dims))
    unstacked = stacked.unstack()
    assert_identical(orig, unstacked.transpose(*dims))