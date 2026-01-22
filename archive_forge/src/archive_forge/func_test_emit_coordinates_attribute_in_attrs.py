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
@pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
def test_emit_coordinates_attribute_in_attrs(self) -> None:
    orig = Dataset({'a': 1, 'b': 1}, coords={'t': np.array('2004-11-01T00:00:00', dtype=np.datetime64)})
    orig['a'].attrs['coordinates'] = None
    enc, _ = conventions.encode_dataset_coordinates(orig)
    assert 'coordinates' not in enc['a'].attrs
    assert 'coordinates' not in enc['a'].encoding
    assert enc['b'].attrs.get('coordinates') == 't'
    assert 'coordinates' not in enc['b'].encoding