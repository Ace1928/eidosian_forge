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
def test_encode_cf_variable_with_vlen_dtype() -> None:
    v = Variable(['x'], np.array(['a', 'b'], dtype=coding.strings.create_vlen_dtype(str)))
    encoded_v = conventions.encode_cf_variable(v)
    assert encoded_v.data.dtype.kind == 'O'
    assert coding.strings.check_vlen_dtype(encoded_v.data.dtype) == str
    v = Variable(['x'], np.array([], dtype=coding.strings.create_vlen_dtype(str)))
    encoded_v = conventions.encode_cf_variable(v)
    assert encoded_v.data.dtype.kind == 'O'
    assert coding.strings.check_vlen_dtype(encoded_v.data.dtype) == str