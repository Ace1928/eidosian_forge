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
def test_groupby_dataset_math_virtual() -> None:
    ds = Dataset({'x': ('t', [1, 2, 3])}, {'t': pd.date_range('20100101', periods=3)})
    grouped = ds.groupby('t.day')
    actual = grouped - grouped.mean(...)
    expected = Dataset({'x': ('t', [0, 0, 0])}, ds[['t', 't.day']])
    assert_identical(actual, expected)