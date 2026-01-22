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
@pytest.mark.parametrize('group', [None, 'group1'])
def test_write_persistence_modes(self, group) -> None:
    original = create_test_data()
    with self.roundtrip(original, save_kwargs={'mode': 'w', 'group': group}, open_kwargs={'group': group}) as actual:
        assert_identical(original, actual)
    with self.roundtrip(original, save_kwargs={'mode': 'w-', 'group': group}, open_kwargs={'group': group}) as actual:
        assert_identical(original, actual)
    with self.create_zarr_target() as store:
        self.save(original, store)
        self.save(original, store, mode='w', group=group)
        with self.open(store, group=group) as actual:
            assert_identical(original, actual)
            with pytest.raises(ValueError):
                self.save(original, store, mode='w-')
    with self.roundtrip(original, save_kwargs={'mode': 'a', 'group': group}, open_kwargs={'group': group}) as actual:
        assert_identical(original, actual)
    ds, ds_to_append, _ = create_append_test_data()
    with self.create_zarr_target() as store_target:
        ds.to_zarr(store_target, mode='w', group=group, **self.version_kwargs)
        ds_to_append.to_zarr(store_target, append_dim='time', group=group, **self.version_kwargs)
        original = xr.concat([ds, ds_to_append], dim='time')
        actual = xr.open_dataset(store_target, group=group, engine='zarr', **self.version_kwargs)
        assert_identical(original, actual)