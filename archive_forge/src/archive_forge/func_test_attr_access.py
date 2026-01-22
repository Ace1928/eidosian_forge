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
def test_attr_access(self) -> None:
    ds = Dataset({'tmin': ('x', [42], {'units': 'Celsius'})}, attrs={'title': 'My test data'})
    assert_identical(ds.tmin, ds['tmin'])
    assert_identical(ds.tmin.x, ds.x)
    assert ds.title == ds.attrs['title']
    assert ds.tmin.units == ds['tmin'].attrs['units']
    assert {'tmin', 'title'} <= set(dir(ds))
    assert 'units' in set(dir(ds.tmin))
    ds.attrs['tmin'] = -999
    assert ds.attrs['tmin'] == -999
    assert_identical(ds.tmin, ds['tmin'])