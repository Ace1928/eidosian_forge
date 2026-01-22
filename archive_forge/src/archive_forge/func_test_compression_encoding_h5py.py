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
def test_compression_encoding_h5py(self) -> None:
    ENCODINGS: tuple[tuple[dict[str, Any], dict[str, Any]], ...] = (({'compression': 'gzip', 'compression_opts': 9}, {'zlib': True, 'complevel': 9}), ({'compression': 'lzf', 'compression_opts': None}, {'compression': 'lzf', 'compression_opts': None}), ({'compression': 'lzf', 'compression_opts': None, 'zlib': True, 'complevel': 9}, {'compression': 'lzf', 'compression_opts': None}))
    for compr_in, compr_out in ENCODINGS:
        data = create_test_data()
        compr_common = {'chunksizes': (5, 5), 'fletcher32': True, 'shuffle': True, 'original_shape': data.var2.shape}
        data['var2'].encoding.update(compr_in)
        data['var2'].encoding.update(compr_common)
        compr_out.update(compr_common)
        data['scalar'] = ('scalar_dim', np.array([2.0]))
        data['scalar'] = data['scalar'][0]
        with self.roundtrip(data) as actual:
            for k, v in compr_out.items():
                assert v == actual['var2'].encoding[k]