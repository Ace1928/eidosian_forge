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
def test_do_not_overwrite_user_coordinates(self) -> None:
    orig = Dataset(coords={'x': [0, 1, 2], 'y': ('x', [5, 6, 7]), 'z': ('x', [8, 9, 10])}, data_vars={'a': ('x', [1, 2, 3]), 'b': ('x', [3, 5, 6])})
    orig['a'].encoding['coordinates'] = 'y'
    orig['b'].encoding['coordinates'] = 'z'
    enc, _ = conventions.encode_dataset_coordinates(orig)
    assert enc['a'].attrs['coordinates'] == 'y'
    assert enc['b'].attrs['coordinates'] == 'z'
    orig['a'].attrs['coordinates'] = 'foo'
    with pytest.raises(ValueError, match="'coordinates' found in both attrs"):
        conventions.encode_dataset_coordinates(orig)