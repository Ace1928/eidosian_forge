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
def test_open_mfdataset_2d(self) -> None:
    original = Dataset({'foo': (['x', 'y'], np.random.randn(10, 8))})
    with create_tmp_file() as tmp1:
        with create_tmp_file() as tmp2:
            with create_tmp_file() as tmp3:
                with create_tmp_file() as tmp4:
                    original.isel(x=slice(5), y=slice(4)).to_netcdf(tmp1)
                    original.isel(x=slice(5, 10), y=slice(4)).to_netcdf(tmp2)
                    original.isel(x=slice(5), y=slice(4, 8)).to_netcdf(tmp3)
                    original.isel(x=slice(5, 10), y=slice(4, 8)).to_netcdf(tmp4)
                    with open_mfdataset([[tmp1, tmp2], [tmp3, tmp4]], combine='nested', concat_dim=['y', 'x']) as actual:
                        assert isinstance(actual.foo.variable.data, da.Array)
                        assert actual.foo.variable.data.chunks == ((5, 5), (4, 4))
                        assert_identical(original, actual)
                    with open_mfdataset([[tmp1, tmp2], [tmp3, tmp4]], combine='nested', concat_dim=['y', 'x'], chunks={'x': 3, 'y': 2}) as actual:
                        assert actual.foo.variable.data.chunks == ((3, 2, 3, 2), (2, 2, 2, 2))