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
def test_display_nbytes() -> None:
    xds = xr.Dataset({'foo': np.arange(1200, dtype=np.int16), 'bar': np.arange(111, dtype=np.int16)})
    actual = repr(xds)
    expected = '\n<xarray.Dataset> Size: 3kB\nDimensions:  (foo: 1200, bar: 111)\nCoordinates:\n  * foo      (foo) int16 2kB 0 1 2 3 4 5 6 ... 1194 1195 1196 1197 1198 1199\n  * bar      (bar) int16 222B 0 1 2 3 4 5 6 7 ... 104 105 106 107 108 109 110\nData variables:\n    *empty*\n    '.strip()
    assert actual == expected
    actual = repr(xds['foo'])
    expected = "\n<xarray.DataArray 'foo' (foo: 1200)> Size: 2kB\narray([   0,    1,    2, ..., 1197, 1198, 1199], dtype=int16)\nCoordinates:\n  * foo      (foo) int16 2kB 0 1 2 3 4 5 6 ... 1194 1195 1196 1197 1198 1199\n".strip()
    assert actual == expected