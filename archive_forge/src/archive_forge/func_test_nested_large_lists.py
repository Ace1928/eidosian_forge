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
def test_nested_large_lists(seq):
    data = [[], [1, 2], None]
    arr = pa.array(seq(data), type=pa.large_list(pa.int16()))
    assert len(arr) == 3
    assert arr.null_count == 1
    assert arr.type == pa.large_list(pa.int16())
    assert arr.to_pylist() == data