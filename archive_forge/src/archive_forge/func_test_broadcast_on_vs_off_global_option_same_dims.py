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
@pytest.mark.parametrize('arithmetic_broadcast', [True, False])
def test_broadcast_on_vs_off_global_option_same_dims(self, arithmetic_broadcast: bool) -> None:
    xda_1 = xr.DataArray([1], dims='x')
    xda_2 = xr.DataArray([1], dims='x')
    expected_xda = xr.DataArray([2.0], dims=('x',))
    with xr.set_options(arithmetic_broadcast=arithmetic_broadcast):
        assert_identical(xda_1 + xda_2, expected_xda)
        assert_identical(xda_1 + np.array([1.0]), expected_xda)
        assert_identical(np.array([1.0]) + xda_1, expected_xda)