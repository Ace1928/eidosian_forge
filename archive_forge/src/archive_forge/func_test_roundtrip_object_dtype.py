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
def test_roundtrip_object_dtype(self) -> None:
    floats = np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=object)
    floats_nans = np.array([np.nan, np.nan, 1.0, 2.0, 3.0], dtype=object)
    bytes_ = np.array([b'ab', b'cdef', b'g'], dtype=object)
    bytes_nans = np.array([b'ab', b'cdef', np.nan], dtype=object)
    strings = np.array(['ab', 'cdef', 'g'], dtype=object)
    strings_nans = np.array(['ab', 'cdef', np.nan], dtype=object)
    all_nans = np.array([np.nan, np.nan], dtype=object)
    original = Dataset({'floats': ('a', floats), 'floats_nans': ('a', floats_nans), 'bytes': ('b', bytes_), 'bytes_nans': ('b', bytes_nans), 'strings': ('b', strings), 'strings_nans': ('b', strings_nans), 'all_nans': ('c', all_nans), 'nan': ([], np.nan)})
    expected = original.copy(deep=True)
    with self.roundtrip(original) as actual:
        try:
            assert_identical(expected, actual)
        except AssertionError:
            expected['bytes_nans'][-1] = b''
            expected['strings_nans'][-1] = ''
            assert_identical(expected, actual)