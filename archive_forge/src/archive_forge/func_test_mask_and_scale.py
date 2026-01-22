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
def test_mask_and_scale(self) -> None:
    with create_tmp_file() as tmp_file:
        with nc4.Dataset(tmp_file, mode='w') as nc:
            nc.createDimension('t', 5)
            nc.createVariable('x', 'int16', ('t',), fill_value=-1)
            v = nc.variables['x']
            v.set_auto_maskandscale(False)
            v.add_offset = 10
            v.scale_factor = 0.1
            v[:] = np.array([-1, -1, 0, 1, 2])
            dtype = type(v.scale_factor)
        with nc4.Dataset(tmp_file, mode='r') as nc:
            expected = np.ma.array([-1, -1, 10, 10.1, 10.2], mask=[True, True, False, False, False])
            actual = nc.variables['x'][:]
            assert_array_equal(expected, actual)
        with open_dataset(tmp_file) as ds:
            expected = create_masked_and_scaled_data(np.dtype(dtype))
            assert_identical(expected, ds)