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
def test_repr_multiindex_long(self) -> None:
    mindex_long = pd.MultiIndex.from_product([['a', 'b', 'c', 'd'], [1, 2, 3, 4, 5, 6, 7, 8]], names=('level_1', 'level_2'))
    mda_long = DataArray(list(range(32)), coords={'x': mindex_long}, dims='x').astype(np.uint64)
    expected = dedent("            <xarray.DataArray (x: 32)> Size: 256B\n            array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],\n                  dtype=uint64)\n            Coordinates:\n              * x        (x) object 256B MultiIndex\n              * level_1  (x) object 256B 'a' 'a' 'a' 'a' 'a' 'a' ... 'd' 'd' 'd' 'd' 'd' 'd'\n              * level_2  (x) int64 256B 1 2 3 4 5 6 7 8 1 2 3 4 ... 5 6 7 8 1 2 3 4 5 6 7 8")
    assert expected == repr(mda_long)