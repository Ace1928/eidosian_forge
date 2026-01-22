from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
def test_groupby_duplicate_coordinate_labels() -> None:
    array = xr.DataArray([1, 2, 3], [('x', [1, 1, 2])])
    expected = xr.DataArray([3, 3], [('x', [1, 2])])
    actual = array.groupby('x').sum()
    assert_equal(expected, actual)