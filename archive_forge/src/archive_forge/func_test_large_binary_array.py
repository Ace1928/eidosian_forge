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
@pytest.mark.large_memory
@pytest.mark.parametrize('ty', [pa.large_binary(), pa.large_string()])
def test_large_binary_array(ty):
    s = b'0123456789abcdefghijklmnopqrstuvwxyz' * 10
    nrepeats = math.ceil((2 ** 32 + 5) / len(s))
    data = [s] * nrepeats
    arr = pa.array(data, type=ty)
    assert isinstance(arr, pa.Array)
    assert arr.type == ty
    assert len(arr) == nrepeats