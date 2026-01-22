from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
@pytest.mark.pandas
def test_to_pandas_float16_list():
    expected = [[np.float16(1)], [np.float16(2)], [np.float16(3)]]
    arr = pa.array(expected)
    result = arr.to_pandas()
    assert result[0].dtype == 'float16'
    assert result.tolist() == expected