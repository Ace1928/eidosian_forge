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
def test_dictionary_is_always_adaptive():
    typ = pa.dictionary(pa.int8(), value_type=pa.int64())
    a = pa.array(range(2 ** 7), type=typ)
    expected = pa.dictionary(pa.int8(), pa.int64())
    assert a.type.equals(expected)
    a = pa.array(range(2 ** 7 + 1), type=typ)
    expected = pa.dictionary(pa.int16(), pa.int64())
    assert a.type.equals(expected)