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
def test_repr_multiindex(self) -> None:
    data = create_test_multiindex()
    expected = dedent("            <xarray.Dataset> Size: 96B\n            Dimensions:  (x: 4)\n            Coordinates:\n              * x        (x) object 32B MultiIndex\n              * level_1  (x) object 32B 'a' 'a' 'b' 'b'\n              * level_2  (x) int64 32B 1 2 1 2\n            Data variables:\n                *empty*")
    actual = '\n'.join((x.rstrip() for x in repr(data).split('\n')))
    print(actual)
    assert expected == actual
    midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=('a_quite_long_level_name', 'level_2'))
    midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
    data = Dataset({}, midx_coords)
    expected = dedent("            <xarray.Dataset> Size: 96B\n            Dimensions:                  (x: 4)\n            Coordinates:\n              * x                        (x) object 32B MultiIndex\n              * a_quite_long_level_name  (x) object 32B 'a' 'a' 'b' 'b'\n              * level_2                  (x) int64 32B 1 2 1 2\n            Data variables:\n                *empty*")
    actual = '\n'.join((x.rstrip() for x in repr(data).split('\n')))
    print(actual)
    assert expected == actual