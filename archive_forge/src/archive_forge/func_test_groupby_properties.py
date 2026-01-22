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
def test_groupby_properties(self) -> None:
    grouped = self.da.groupby('abc')
    expected_groups = {'a': range(0, 9), 'c': [9], 'b': range(10, 20)}
    assert expected_groups.keys() == grouped.groups.keys()
    for key in expected_groups:
        expected_group = expected_groups[key]
        actual_group = grouped.groups[key]
        assert not isinstance(expected_group, slice)
        assert not isinstance(actual_group, slice)
        np.testing.assert_array_equal(expected_group, actual_group)
    assert 3 == len(grouped)