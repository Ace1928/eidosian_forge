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
@pytest.mark.parametrize(['combine_attrs', 'attrs', 'expected', 'expect_error'], (pytest.param('drop', [{'a': 1}, {'a': 2}], {}, False, id='drop'), pytest.param('override', [{'a': 1}, {'a': 2}], {'a': 1}, False, id='override'), pytest.param('no_conflicts', [{'a': 1}, {'a': 2}], None, True, id='no_conflicts'), pytest.param('identical', [{'a': 1, 'b': 2}, {'a': 1, 'c': 3}], None, True, id='identical'), pytest.param('drop_conflicts', [{'a': 1, 'b': 2}, {'b': -1, 'c': 3}], {'a': 1, 'c': 3}, False, id='drop_conflicts')))
def test_open_mfdataset_dataset_combine_attrs(self, combine_attrs, attrs, expected, expect_error):
    with self.setup_files_and_datasets() as (files, [ds1, ds2]):
        for i, f in enumerate(files):
            ds = open_dataset(f).load()
            ds.attrs = attrs[i]
            ds.close()
            ds.to_netcdf(f)
        if expect_error:
            with pytest.raises(xr.MergeError):
                xr.open_mfdataset(files, combine='nested', concat_dim='t', combine_attrs=combine_attrs)
        else:
            with xr.open_mfdataset(files, combine='nested', concat_dim='t', combine_attrs=combine_attrs) as ds:
                assert ds.attrs == expected