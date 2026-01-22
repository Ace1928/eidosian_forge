from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_modify_inplace(self) -> None:
    a = Dataset()
    vec = np.random.random((10,))
    attributes = {'foo': 'bar'}
    a['x'] = ('x', vec, attributes)
    assert 'x' in a.coords
    assert isinstance(a.coords['x'].to_index(), pd.Index)
    assert_identical(a.coords['x'].variable, a.variables['x'])
    b = Dataset()
    b['x'] = ('x', vec, attributes)
    assert_identical(a['x'], b['x'])
    assert a.sizes == b.sizes
    a['x'] = ('x', vec[:5])
    a['z'] = ('x', np.arange(5))
    with pytest.raises(ValueError):
        a['x'] = ('x', vec[:4])
    arr = np.random.random((10, 1))
    scal = np.array(0)
    with pytest.raises(ValueError):
        a['y'] = ('y', arr)
    with pytest.raises(ValueError):
        a['y'] = ('y', scal)
    assert 'y' not in a.dims