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
def test_struct_array_dims(self) -> None:
    """
        This test checks subtraction of two DataArrays for the case
        when dimension is a structured array.
        """
    p_data = np.array([('Abe', 180), ('Stacy', 150), ('Dick', 200)], dtype=[('name', '|S256'), ('height', object)])
    weights_0 = DataArray([80, 56, 120], dims=['participant'], coords={'participant': p_data})
    weights_1 = DataArray([81, 52, 115], dims=['participant'], coords={'participant': p_data})
    actual = weights_1 - weights_0
    expected = DataArray([1, -4, -5], dims=['participant'], coords={'participant': p_data})
    assert_identical(actual, expected)
    p_data_alt = np.array([('Abe', 180), ('Stacy', 151), ('Dick', 200)], dtype=[('name', '|S256'), ('height', object)])
    weights_1 = DataArray([81, 52, 115], dims=['participant'], coords={'participant': p_data_alt})
    actual = weights_1 - weights_0
    expected = DataArray([1, -5], dims=['participant'], coords={'participant': p_data[[0, 2]]})
    assert_identical(actual, expected)
    p_data_nan = np.array([('Abe', 180), ('Stacy', np.nan), ('Dick', 200)], dtype=[('name', '|S256'), ('height', object)])
    weights_1 = DataArray([81, 52, 115], dims=['participant'], coords={'participant': p_data_nan})
    actual = weights_1 - weights_0
    expected = DataArray([1, -5], dims=['participant'], coords={'participant': p_data[[0, 2]]})
    assert_identical(actual, expected)