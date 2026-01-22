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
def test_groupby_multidim_map(self) -> None:
    array = self.make_groupby_multidim_example_array()
    actual = array.groupby('lon').map(lambda x: x - x.mean())
    expected = DataArray([[[-2.5, -6.0], [-5.0, -8.5]], [[2.5, 3.0], [8.0, 8.5]]], coords=array.coords, dims=array.dims)
    assert_identical(expected, actual)