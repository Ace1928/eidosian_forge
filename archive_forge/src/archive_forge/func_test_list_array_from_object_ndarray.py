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
@pytest.mark.parametrize(('data', 'value_type'), [([True, False], pa.bool_()), ([None, None], pa.null()), ([1, 2, None], pa.int8()), ([1, 2.0, 3.0, None], pa.float32()), ([datetime.date.today(), None], pa.date32()), ([None, datetime.date.today()], pa.date64()), ([datetime.time(1, 1, 1), None], pa.time32('s')), ([None, datetime.time(2, 2, 2)], pa.time64('us')), ([datetime.datetime.now(), None], pa.timestamp('us')), ([datetime.timedelta(seconds=10)], pa.duration('s')), ([b'a', b'b'], pa.binary()), ([b'aaa', b'bbb', b'ccc'], pa.binary(3)), ([b'a', b'b', b'c'], pa.large_binary()), (['a', 'b', 'c'], pa.string()), (['a', 'b', 'c'], pa.large_string()), ([{'a': 1, 'b': 2}, None, {'a': 5, 'b': None}], pa.struct([('a', pa.int8()), ('b', pa.int16())]))])
def test_list_array_from_object_ndarray(data, value_type):
    ty = pa.list_(value_type)
    ndarray = np.array(data, dtype=object)
    arr = pa.array([ndarray], type=ty)
    assert arr.type.equals(ty)
    assert arr.to_pylist() == [data]