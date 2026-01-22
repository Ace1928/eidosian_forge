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
def test_to_dataset_split(self) -> None:
    array = DataArray([[1, 2], [3, 4], [5, 6]], coords=[('x', list('abc')), ('y', [0.0, 0.1])], attrs={'a': 1})
    expected = Dataset({'a': ('y', [1, 2]), 'b': ('y', [3, 4]), 'c': ('y', [5, 6])}, coords={'y': [0.0, 0.1]}, attrs={'a': 1})
    actual = array.to_dataset('x')
    assert_identical(expected, actual)
    with pytest.raises(TypeError):
        array.to_dataset('x', name='foo')
    roundtripped = actual.to_dataarray(dim='x')
    assert_identical(array, roundtripped)
    array = DataArray([1, 2, 3], dims='x')
    expected = Dataset({0: 1, 1: 2, 2: 3})
    actual = array.to_dataset('x')
    assert_identical(expected, actual)