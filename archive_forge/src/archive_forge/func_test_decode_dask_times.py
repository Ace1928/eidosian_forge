from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from xarray import (
from xarray.backends.common import WritableCFDataStore
from xarray.backends.memory import InMemoryDataStore
from xarray.conventions import decode_cf
from xarray.testing import assert_identical
from xarray.tests import (
from xarray.tests.test_backends import CFEncodedBase
@requires_dask
def test_decode_dask_times(self) -> None:
    original = Dataset.from_dict({'coords': {}, 'dims': {'time': 5}, 'data_vars': {'average_T1': {'dims': ('time',), 'attrs': {'units': 'days since 1958-01-01 00:00:00'}, 'data': [89289.0, 88024.0, 88389.0, 88754.0, 89119.0]}}})
    assert_identical(conventions.decode_cf(original.chunk()), conventions.decode_cf(original).chunk())