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
@parametrize_with_sequence_types
def test_sequence_custom_integers(seq):
    expected = [0, 42, 2 ** 33 + 1, -2 ** 63]
    data = list(map(MyInt, expected))
    arr = pa.array(seq(data), type=pa.int64())
    assert arr.to_pylist() == expected