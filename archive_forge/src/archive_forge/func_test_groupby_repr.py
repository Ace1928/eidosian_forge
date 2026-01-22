from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
@pytest.mark.parametrize('dim', ['x', 'y', 'z', 'month'])
@pytest.mark.parametrize('obj', [repr_da, repr_da.to_dataset(name='a')])
def test_groupby_repr(obj, dim) -> None:
    actual = repr(obj.groupby(dim))
    expected = f'{obj.__class__.__name__}GroupBy'
    expected += ', grouped over %r' % dim
    expected += '\n%r groups with labels ' % len(np.unique(obj[dim]))
    if dim == 'x':
        expected += '1, 2, 3, 4, 5.'
    elif dim == 'y':
        expected += '0, 1, 2, 3, 4, 5, ..., 15, 16, 17, 18, 19.'
    elif dim == 'z':
        expected += "'a', 'b', 'c'."
    elif dim == 'month':
        expected += '1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12.'
    assert actual == expected