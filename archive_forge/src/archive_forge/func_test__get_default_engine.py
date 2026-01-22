from __future__ import annotations
from numbers import Number
import numpy as np
import pytest
import xarray as xr
from xarray.backends.api import _get_default_engine
from xarray.tests import (
@requires_netCDF4
@requires_scipy
def test__get_default_engine() -> None:
    engine_remote = _get_default_engine('http://example.org/test.nc', allow_remote=True)
    assert engine_remote == 'netcdf4'
    engine_gz = _get_default_engine('/example.gz')
    assert engine_gz == 'scipy'
    engine_default = _get_default_engine('/example')
    assert engine_default == 'netcdf4'