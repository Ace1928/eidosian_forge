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
@requires_dask
@pytest.mark.skipif(ON_WINDOWS, reason='Very flaky on Windows CI. Can re-enable assuming it starts consistently passing.')
def test_chunk_encoding_with_dask(self) -> None:
    ds = xr.DataArray(np.arange(12), dims='x', name='var1').to_dataset()
    ds_chunk4 = ds.chunk({'x': 4})
    with self.roundtrip(ds_chunk4) as actual:
        assert (4,) == actual['var1'].encoding['chunks']
    ds_chunk_irreg = ds.chunk({'x': (5, 4, 3)})
    with pytest.raises(ValueError, match='uniform chunk sizes.'):
        with self.roundtrip(ds_chunk_irreg) as actual:
            pass
    badenc = ds.chunk({'x': 4})
    badenc.var1.encoding['chunks'] = (6,)
    with pytest.raises(ValueError, match="named 'var1' would overlap"):
        with self.roundtrip(badenc) as actual:
            pass
    with self.roundtrip(badenc, save_kwargs={'safe_chunks': False}) as actual:
        pass
    goodenc = ds.chunk({'x': 4})
    goodenc.var1.encoding['chunks'] = (2,)
    with self.roundtrip(goodenc) as actual:
        pass
    goodenc = ds.chunk({'x': (3, 3, 6)})
    goodenc.var1.encoding['chunks'] = (3,)
    with self.roundtrip(goodenc) as actual:
        pass
    goodenc = ds.chunk({'x': (3, 6, 3)})
    goodenc.var1.encoding['chunks'] = (3,)
    with self.roundtrip(goodenc) as actual:
        pass
    ds_chunk_irreg = ds.chunk({'x': (5, 5, 2)})
    with self.roundtrip(ds_chunk_irreg) as actual:
        assert (5,) == actual['var1'].encoding['chunks']
    with self.roundtrip(ds_chunk_irreg) as original:
        with self.roundtrip(original) as actual:
            assert_identical(original, actual)
    badenc = ds.chunk({'x': (3, 5, 3, 1)})
    badenc.var1.encoding['chunks'] = (3,)
    with pytest.raises(ValueError, match='would overlap multiple dask chunks'):
        with self.roundtrip(badenc) as actual:
            pass
    for chunk_enc in (4, (4,)):
        ds_chunk4['var1'].encoding.update({'chunks': chunk_enc})
        with self.roundtrip(ds_chunk4) as actual:
            assert (4,) == actual['var1'].encoding['chunks']
    ds_chunk4['var1'].encoding.update({'chunks': 5})
    with pytest.raises(ValueError, match="named 'var1' would overlap"):
        with self.roundtrip(ds_chunk4) as actual:
            pass
    with self.roundtrip(ds_chunk4, save_kwargs={'safe_chunks': False}) as actual:
        pass