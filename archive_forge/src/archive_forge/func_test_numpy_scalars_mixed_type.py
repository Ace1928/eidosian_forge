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
def test_numpy_scalars_mixed_type():
    data = [np.int32(10), np.float32(0.5)]
    arr = pa.array(data)
    expected = pa.array([10, 0.5], type='float64')
    assert arr.equals(expected)
    data = [np.int8(10), np.float32(0.5)]
    arr = pa.array(data)
    expected = pa.array([10, 0.5], type='float32')
    assert arr.equals(expected)