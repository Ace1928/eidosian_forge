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
@requires_h5netcdf
@requires_netCDF4
def test_load_single_value_h5netcdf(tmp_path: Path) -> None:
    """Test that numeric single-element vector attributes are handled fine.

    At present (h5netcdf v0.8.1), the h5netcdf exposes single-valued numeric variable
    attributes as arrays of length 1, as opposed to scalars for the NetCDF4
    backend.  This was leading to a ValueError upon loading a single value from
    a file, see #4471.  Test that loading causes no failure.
    """
    ds = xr.Dataset({'test': xr.DataArray(np.array([0]), dims=('x',), attrs={'scale_factor': 1, 'add_offset': 0})})
    ds.to_netcdf(tmp_path / 'test.nc')
    with xr.open_dataset(tmp_path / 'test.nc', engine='h5netcdf') as ds2:
        ds2['test'][0].load()