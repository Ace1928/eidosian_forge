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
def test_dataset_math(self) -> None:
    obs = Dataset({'tmin': ('x', np.arange(5)), 'tmax': ('x', 10 + np.arange(5))}, {'x': ('x', 0.5 * np.arange(5)), 'loc': ('x', range(-2, 3))})
    actual1 = 2 * obs['tmax']
    expected1 = DataArray(2 * (10 + np.arange(5)), obs.coords, name='tmax')
    assert_identical(actual1, expected1)
    actual2 = obs['tmax'] - obs['tmin']
    expected2 = DataArray(10 * np.ones(5), obs.coords)
    assert_identical(actual2, expected2)
    sim = Dataset({'tmin': ('x', 1 + np.arange(5)), 'tmax': ('x', 11 + np.arange(5)), 'x': ('x', 0.5 * np.arange(5))})
    actual3 = sim['tmin'] - obs['tmin']
    expected3 = DataArray(np.ones(5), obs.coords, name='tmin')
    assert_identical(actual3, expected3)
    actual4 = -obs['tmin'] + sim['tmin']
    assert_identical(actual4, expected3)
    actual5 = sim['tmin'].copy()
    actual5 -= obs['tmin']
    assert_identical(actual5, expected3)
    actual6 = sim.copy()
    actual6['tmin'] = sim['tmin'] - obs['tmin']
    expected6 = Dataset({'tmin': ('x', np.ones(5)), 'tmax': ('x', sim['tmax'].values)}, obs.coords)
    assert_identical(actual6, expected6)
    actual7 = sim.copy()
    actual7['tmin'] -= obs['tmin']
    assert_identical(actual7, expected6)