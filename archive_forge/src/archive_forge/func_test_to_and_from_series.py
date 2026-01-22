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
def test_to_and_from_series(self) -> None:
    expected = self.dv.to_dataframe()['foo']
    actual = self.dv.to_series()
    assert_array_equal(expected.values, actual.values)
    assert_array_equal(expected.index.values, actual.index.values)
    assert 'foo' == actual.name
    assert_identical(self.dv, DataArray.from_series(actual).drop_vars(['x', 'y']))
    actual.name = None
    expected_da = self.dv.rename(None)
    assert_identical(expected_da, DataArray.from_series(actual).drop_vars(['x', 'y']))