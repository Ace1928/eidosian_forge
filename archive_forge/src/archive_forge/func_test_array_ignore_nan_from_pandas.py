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
def test_array_ignore_nan_from_pandas():
    with pytest.raises(ValueError):
        pa.array([np.nan, 'str'])
    arr = pa.array([np.nan, 'str'], from_pandas=True)
    expected = pa.array([None, 'str'])
    assert arr.equals(expected)