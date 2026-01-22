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
@pytest.mark.parametrize('compression', [None, 'zlib', 'szip', 'zstd', 'blosc_lz', 'blosc_lz4', 'blosc_lz4hc', 'blosc_zlib', 'blosc_zstd'])
@requires_netCDF4_1_6_2_or_above
@pytest.mark.xfail(ON_WINDOWS, reason='new compression not yet implemented')
def test_compression_encoding(self, compression: str | None) -> None:
    data = create_test_data(dim_sizes=(20, 80, 10))
    encoding_params: dict[str, Any] = dict(compression=compression, blosc_shuffle=1)
    data['var2'].encoding.update(encoding_params)
    data['var2'].encoding.update({'chunksizes': (20, 40), 'original_shape': data.var2.shape, 'blosc_shuffle': 1, 'fletcher32': False})
    with self.roundtrip(data) as actual:
        expected_encoding = data['var2'].encoding.copy()
        compression = expected_encoding.pop('compression')
        blosc_shuffle = expected_encoding.pop('blosc_shuffle')
        if compression is not None:
            if 'blosc' in compression and blosc_shuffle:
                expected_encoding['blosc'] = {'compressor': compression, 'shuffle': blosc_shuffle}
                expected_encoding['shuffle'] = False
            elif compression == 'szip':
                expected_encoding['szip'] = {'coding': 'nn', 'pixels_per_block': 8}
                expected_encoding['shuffle'] = False
            else:
                expected_encoding[compression] = True
                if compression == 'zstd':
                    expected_encoding['shuffle'] = False
        else:
            expected_encoding['shuffle'] = False
        actual_encoding = actual['var2'].encoding
        assert expected_encoding.items() <= actual_encoding.items()
    if encoding_params['compression'] is not None and 'blosc' not in encoding_params['compression']:
        expected = data.isel(dim1=0)
        with self.roundtrip(expected) as actual:
            assert_equal(expected, actual)