from __future__ import annotations
import sys
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray
import xarray as xr
from xarray.core import formatting
from xarray.tests import requires_cftime, requires_dask, requires_netCDF4
@pytest.mark.filterwarnings('error')
def test_diff_attrs_repr_with_array(self) -> None:
    attrs_a = {'attr': np.array([0, 1])}
    attrs_b = {'attr': 1}
    expected = dedent('            Differing attributes:\n            L   attr: [0 1]\n            R   attr: 1\n            ').strip()
    actual = formatting.diff_attrs_repr(attrs_a, attrs_b, 'equals')
    assert expected == actual
    attrs_c = {'attr': np.array([-3, 5])}
    expected = dedent('            Differing attributes:\n            L   attr: [0 1]\n            R   attr: [-3  5]\n            ').strip()
    actual = formatting.diff_attrs_repr(attrs_a, attrs_c, 'equals')
    assert expected == actual
    attrs_c = {'attr': np.array([0, 1, 2])}
    expected = dedent('            Differing attributes:\n            L   attr: [0 1]\n            R   attr: [0 1 2]\n            ').strip()
    actual = formatting.diff_attrs_repr(attrs_a, attrs_c, 'equals')
    assert expected == actual