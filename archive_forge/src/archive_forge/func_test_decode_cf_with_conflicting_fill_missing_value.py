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
def test_decode_cf_with_conflicting_fill_missing_value() -> None:
    expected = Variable(['t'], [np.nan, np.nan, 2], {'units': 'foobar'})
    var = Variable(['t'], np.arange(3), {'units': 'foobar', 'missing_value': 0, '_FillValue': 1})
    with pytest.warns(SerializationWarning, match='has multiple fill'):
        actual = conventions.decode_cf_variable('t', var)
        assert_identical(actual, expected)
    expected = Variable(['t'], np.arange(10), {'units': 'foobar'})
    var = Variable(['t'], np.arange(10), {'units': 'foobar', 'missing_value': np.nan, '_FillValue': np.nan})
    with pytest.warns(SerializationWarning) as winfo:
        actual = conventions.decode_cf_variable('t', var)
    for aw in winfo:
        assert 'non-conforming' in str(aw.message)
    assert_identical(actual, expected)
    var = Variable(['t'], np.arange(10), {'units': 'foobar', 'missing_value': np.float32(np.nan), '_FillValue': np.float32(np.nan)})
    with pytest.warns(SerializationWarning) as winfo:
        actual = conventions.decode_cf_variable('t', var)
    for aw in winfo:
        assert 'non-conforming' in str(aw.message)
    assert_identical(actual, expected)