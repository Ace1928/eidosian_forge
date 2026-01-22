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
@pytest.mark.parametrize('consolidated', [False, True, None])
@pytest.mark.parametrize('compute', [False, True])
@pytest.mark.parametrize('use_dask', [False, True])
@pytest.mark.parametrize('write_empty', [False, True, None])
def test_write_region(self, consolidated, compute, use_dask, write_empty) -> None:
    if (use_dask or not compute) and (not has_dask):
        pytest.skip('requires dask')
    if consolidated and self.zarr_version > 2:
        pytest.xfail('consolidated metadata is not supported for zarr v3 yet')
    zeros = Dataset({'u': (('x',), np.zeros(10))})
    nonzeros = Dataset({'u': (('x',), np.arange(1, 11))})
    if use_dask:
        zeros = zeros.chunk(2)
        nonzeros = nonzeros.chunk(2)
    with self.create_zarr_target() as store:
        zeros.to_zarr(store, consolidated=consolidated, compute=compute, encoding={'u': dict(chunks=2)}, **self.version_kwargs)
        if compute:
            with xr.open_zarr(store, consolidated=consolidated, **self.version_kwargs) as actual:
                assert_identical(actual, zeros)
        for i in range(0, 10, 2):
            region = {'x': slice(i, i + 2)}
            nonzeros.isel(region).to_zarr(store, region=region, consolidated=consolidated, write_empty_chunks=write_empty, **self.version_kwargs)
        with xr.open_zarr(store, consolidated=consolidated, **self.version_kwargs) as actual:
            assert_identical(actual, nonzeros)