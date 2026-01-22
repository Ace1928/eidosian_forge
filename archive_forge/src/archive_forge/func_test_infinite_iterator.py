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
def test_infinite_iterator():
    expected = pa.array((0, 1, 2))
    arr1 = pa.array(itertools.count(0), size=3)
    assert arr1.equals(expected)
    arr1 = pa.array(itertools.count(0), type=pa.int64(), size=3)
    assert arr1.equals(expected)