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
def test_double_auto_coerce_from_integer():
    data = [1.5, 1.0, None, 2.5, None, None]
    arr = pa.array(data)
    data2 = [1.5, 1, None, 2.5, None, None]
    arr2 = pa.array(data2)
    assert arr.equals(arr2)
    data3 = [1, 1.5, None, 2.5, None, None]
    arr3 = pa.array(data3)
    data4 = [1.0, 1.5, None, 2.5, None, None]
    arr4 = pa.array(data4)
    assert arr3.equals(arr4)