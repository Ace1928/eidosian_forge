from __future__ import annotations
import pickle
from typing import TYPE_CHECKING, Any
import numpy as np
import pytest
from dask.distributed import Client, Lock
from distributed.client import futures_of
from distributed.utils_test import (  # noqa: F401
import xarray as xr
from xarray.backends.locks import HDF5_LOCK, CombinedLock, SerializableLock
from xarray.tests import (
from xarray.tests.test_backends import (
from xarray.tests.test_dataset import create_test_data
@requires_netCDF4
def test_dask_distributed_write_netcdf_with_dimensionless_variables(loop, tmp_netcdf_filename):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            original = xr.Dataset({'x': da.zeros(())})
            original.to_netcdf(tmp_netcdf_filename)
            with xr.open_dataset(tmp_netcdf_filename) as actual:
                assert actual.x.shape == ()