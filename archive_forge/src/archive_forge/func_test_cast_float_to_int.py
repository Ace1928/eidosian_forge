import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_cast_float_to_int():
    float_scalar = pa.scalar(1.5, type=pa.float64())
    unsafe_cast = float_scalar.cast(pa.int64(), safe=False)
    expected_unsafe_cast = pa.scalar(1, type=pa.int64())
    assert unsafe_cast == expected_unsafe_cast
    with pytest.raises(pa.ArrowInvalid):
        float_scalar.cast(pa.int64())