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
def test_struct_from_dicts():
    ty = pa.struct([pa.field('a', pa.int32()), pa.field('b', pa.string()), pa.field('c', pa.bool_())])
    arr = pa.array([], type=ty)
    assert arr.to_pylist() == []
    data = [{'a': 5, 'b': 'foo', 'c': True}, {'a': 6, 'b': 'bar', 'c': False}]
    arr = pa.array(data, type=ty)
    assert arr.to_pylist() == data
    data = [{'a': 5, 'c': True}, None, {}, {'a': None, 'b': 'bar'}]
    arr = pa.array(data, type=ty)
    expected = [{'a': 5, 'b': None, 'c': True}, None, {'a': None, 'b': None, 'c': None}, {'a': None, 'b': 'bar', 'c': None}]
    assert arr.to_pylist() == expected