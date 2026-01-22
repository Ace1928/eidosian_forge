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
@requires_iris
@pytest.mark.parametrize('coord_values', [['IA', 'IL', 'IN'], [0, 2, 1]])
def test_fallback_to_iris_AuxCoord(self, coord_values) -> None:
    from iris.coords import AuxCoord
    from iris.cube import Cube
    data = [0, 0, 0]
    da = xr.DataArray(data, coords=[coord_values], dims=['space'])
    result = xr.DataArray.to_iris(da)
    expected = Cube(data, aux_coords_and_dims=[(AuxCoord(coord_values, var_name='space'), 0)])
    assert result == expected