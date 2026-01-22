import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_cast_int_to_float():
    int_scalar = pa.scalar(18014398509481983, type=pa.int64())
    unsafe_cast = int_scalar.cast(pa.float64(), safe=False)
    expected_unsafe_cast = pa.scalar(1.8014398509481984e+16, type=pa.float64())
    assert unsafe_cast == expected_unsafe_cast
    with pytest.raises(pa.ArrowInvalid):
        int_scalar.cast(pa.float64())