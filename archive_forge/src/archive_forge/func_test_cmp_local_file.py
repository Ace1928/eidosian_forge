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
def test_cmp_local_file(self) -> None:
    with self.create_datasets() as (actual, expected):
        assert_equal(actual, expected)
        assert 'NC_GLOBAL' not in actual.attrs
        assert 'history' in actual.attrs
        assert actual.attrs.keys() == expected.attrs.keys()
    with self.create_datasets() as (actual, expected):
        assert_equal(actual[{'l': 2}], expected[{'l': 2}])
    with self.create_datasets() as (actual, expected):
        assert_equal(actual.isel(i=0, j=-1), expected.isel(i=0, j=-1))
    with self.create_datasets() as (actual, expected):
        assert_equal(actual.isel(j=slice(1, 2)), expected.isel(j=slice(1, 2)))
    with self.create_datasets() as (actual, expected):
        indexers = {'i': [1, 0, 0], 'j': [1, 2, 0, 1]}
        assert_equal(actual.isel(**indexers), expected.isel(**indexers))
    with self.create_datasets() as (actual, expected):
        indexers2 = {'i': DataArray([0, 1, 0], dims='a'), 'j': DataArray([0, 2, 1], dims='a')}
        assert_equal(actual.isel(**indexers2), expected.isel(**indexers2))