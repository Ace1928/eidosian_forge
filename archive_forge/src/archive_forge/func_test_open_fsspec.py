from __future__ import annotations
import contextlib
import gzip
import itertools
import math
import os.path
import pickle
import platform
import re
import shutil
import sys
import tempfile
import uuid
import warnings
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack
from io import BytesIO
from os import listdir
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.errors import OutOfBoundsDatetime
import xarray as xr
from xarray import (
from xarray.backends.common import robust_getitem
from xarray.backends.h5netcdf_ import H5netcdfBackendEntrypoint
from xarray.backends.netcdf3 import _nc3_dtype_coercions
from xarray.backends.netCDF4_ import (
from xarray.backends.pydap_ import PydapDataStore
from xarray.backends.scipy_ import ScipyBackendEntrypoint
from xarray.coding.cftime_offsets import cftime_range
from xarray.coding.strings import check_vlen_dtype, create_vlen_dtype
from xarray.coding.variables import SerializationWarning
from xarray.conventions import encode_dataset_coordinates
from xarray.core import indexing
from xarray.core.options import set_options
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_coding_times import (
from xarray.tests.test_dataset import (
@requires_zarr
@requires_fsspec
@pytest.mark.filterwarnings('ignore:deallocating CachingFileManager')
def test_open_fsspec() -> None:
    import fsspec
    import zarr
    if not hasattr(zarr.storage, 'FSStore') or not hasattr(zarr.storage.FSStore, 'getitems'):
        pytest.skip('zarr too old')
    ds = open_dataset(os.path.join(os.path.dirname(__file__), 'data', 'example_1.nc'))
    m = fsspec.filesystem('memory')
    mm = m.get_mapper('out1.zarr')
    ds.to_zarr(mm)
    ds0 = ds.copy()
    ds0['time'] = ds.time + pd.to_timedelta('1 day')
    mm = m.get_mapper('out2.zarr')
    ds0.to_zarr(mm)
    url = 'memory://out2.zarr'
    ds2 = open_dataset(url, engine='zarr')
    xr.testing.assert_equal(ds0, ds2)
    url = 'simplecache::memory://out2.zarr'
    ds2 = open_dataset(url, engine='zarr')
    xr.testing.assert_equal(ds0, ds2)
    if has_dask:
        url = 'memory://out*.zarr'
        ds2 = open_mfdataset(url, engine='zarr')
        xr.testing.assert_equal(xr.concat([ds, ds0], dim='time'), ds2)
        url = 'simplecache::memory://out*.zarr'
        ds2 = open_mfdataset(url, engine='zarr')
        xr.testing.assert_equal(xr.concat([ds, ds0], dim='time'), ds2)