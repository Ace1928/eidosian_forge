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
def test_setncattr_string(self) -> None:
    list_of_strings = ['list', 'of', 'strings']
    one_element_list_of_strings = ['one element']
    one_string = 'one string'
    attrs = {'foo': list_of_strings, 'bar': one_element_list_of_strings, 'baz': one_string}
    ds = Dataset({'x': ('y', [1, 2, 3], attrs)}, attrs=attrs)
    with self.roundtrip(ds) as actual:
        for totest in [actual, actual['x']]:
            assert_array_equal(list_of_strings, totest.attrs['foo'])
            assert_array_equal(one_element_list_of_strings, totest.attrs['bar'])
            assert one_string == totest.attrs['baz']