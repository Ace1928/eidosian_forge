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
@pytest.mark.parametrize('decoded_fn, encoded_fn', [(create_unsigned_masked_scaled_data, create_encoded_unsigned_masked_scaled_data), pytest.param(create_bad_unsigned_masked_scaled_data, create_bad_encoded_unsigned_masked_scaled_data, marks=pytest.mark.xfail(reason='Bad _Unsigned attribute.')), (create_signed_masked_scaled_data, create_encoded_signed_masked_scaled_data), (create_masked_and_scaled_data, create_encoded_masked_and_scaled_data)])
@pytest.mark.parametrize('dtype', [np.dtype('float64'), np.dtype('float32')])
def test_roundtrip_mask_and_scale(self, decoded_fn, encoded_fn, dtype) -> None:
    if hasattr(self, 'zarr_version') and dtype == np.float32:
        pytest.skip('float32 will be treated as float64 in zarr')
    decoded = decoded_fn(dtype)
    encoded = encoded_fn(dtype)
    with self.roundtrip(decoded) as actual:
        for k in decoded.variables:
            assert decoded.variables[k].dtype == actual.variables[k].dtype
        assert_allclose(decoded, actual, decode_bytes=False)
    with self.roundtrip(decoded, open_kwargs=dict(decode_cf=False)) as actual:
        for k in encoded.variables:
            assert encoded.variables[k].dtype == actual.variables[k].dtype
        assert_allclose(encoded, actual, decode_bytes=False)
    with self.roundtrip(encoded, open_kwargs=dict(decode_cf=False)) as actual:
        for k in encoded.variables:
            assert encoded.variables[k].dtype == actual.variables[k].dtype
        assert_allclose(encoded, actual, decode_bytes=False)
    assert_allclose(encoded, encoded_fn(dtype), decode_bytes=False)
    with self.roundtrip(encoded) as actual:
        for k in decoded.variables:
            assert decoded.variables[k].dtype == actual.variables[k].dtype
        assert_allclose(decoded, actual, decode_bytes=False)