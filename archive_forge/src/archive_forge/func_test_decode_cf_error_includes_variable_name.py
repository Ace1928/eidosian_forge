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
def test_decode_cf_error_includes_variable_name():
    ds = Dataset({'invalid': ([], 1e+36, {'units': 'days since 2000-01-01'})})
    with pytest.raises(ValueError, match="Failed to decode variable 'invalid'"):
        decode_cf(ds)