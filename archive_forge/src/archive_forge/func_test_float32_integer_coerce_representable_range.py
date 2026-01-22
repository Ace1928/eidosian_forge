import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
def test_float32_integer_coerce_representable_range():
    f32 = np.float32
    valid_values = [f32(1.5), 1 << 24, -(1 << 24)]
    invalid_values = [f32(1.5), (1 << 24) + 1]
    invalid_values2 = [f32(1.5), -((1 << 24) + 1)]
    pa.array(valid_values, type=pa.float32())
    with pytest.raises(ValueError):
        pa.array(invalid_values, type=pa.float32())
    with pytest.raises(ValueError):
        pa.array(invalid_values2, type=pa.float32())