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
@parametrize_with_collections_types
@pytest.mark.parametrize(('data', 'scalar_data', 'value_type'), [([True, False, None], [pa.scalar(True), pa.scalar(False), None], pa.bool_()), ([1, 2, None], [pa.scalar(1), pa.scalar(2), pa.scalar(None, pa.int64())], pa.int64()), ([1, None, None], [pa.scalar(1), None, pa.scalar(None, pa.int64())], pa.int64()), ([None, None], [pa.scalar(None), pa.scalar(None)], pa.null()), ([1.0, 2.0, None], [pa.scalar(1.0), pa.scalar(2.0), None], pa.float64()), ([None, datetime.date.today()], [None, pa.scalar(datetime.date.today())], pa.date32()), ([None, datetime.date.today()], [None, pa.scalar(datetime.date.today(), pa.date64())], pa.date64()), ([datetime.time(1, 1, 1), None], [pa.scalar(datetime.time(1, 1, 1)), None], pa.time64('us')), ([datetime.timedelta(seconds=10)], [pa.scalar(datetime.timedelta(seconds=10))], pa.duration('us')), ([None, datetime.datetime(2014, 1, 1)], [None, pa.scalar(datetime.datetime(2014, 1, 1))], pa.timestamp('us')), ([pa.MonthDayNano([1, -1, -10100])], [pa.scalar(pa.MonthDayNano([1, -1, -10100]))], pa.month_day_nano_interval()), (['a', 'b'], [pa.scalar('a'), pa.scalar('b')], pa.string()), ([b'a', b'b'], [pa.scalar(b'a'), pa.scalar(b'b')], pa.binary()), ([b'a', b'b'], [pa.scalar(b'a', pa.binary(1)), pa.scalar(b'b', pa.binary(1))], pa.binary(1)), ([[1, 2, 3]], [pa.scalar([1, 2, 3])], pa.list_(pa.int64())), ([['a', 'b']], [pa.scalar(['a', 'b'])], pa.list_(pa.string())), ([1, 2, None], [pa.scalar(1, type=pa.int8()), pa.scalar(2, type=pa.int8()), None], pa.int8()), ([1, None], [pa.scalar(1.0, type=pa.int32()), None], pa.int32()), (['aaa', 'bbb'], [pa.scalar('aaa', type=pa.binary(3)), pa.scalar('bbb', type=pa.binary(3))], pa.binary(3)), ([b'a'], [pa.scalar('a', type=pa.large_binary())], pa.large_binary()), (['a'], [pa.scalar('a', type=pa.large_string())], pa.large_string()), (['a'], [pa.scalar('a', type=pa.dictionary(pa.int64(), pa.string()))], pa.dictionary(pa.int64(), pa.string())), (['a', 'b'], [pa.scalar('a', pa.dictionary(pa.int64(), pa.string())), pa.scalar('b', pa.dictionary(pa.int64(), pa.string()))], pa.dictionary(pa.int64(), pa.string())), ([1], [pa.scalar(1, type=pa.dictionary(pa.int64(), pa.int32()))], pa.dictionary(pa.int64(), pa.int32())), ([(1, 2)], [pa.scalar([('a', 1), ('b', 2)], type=pa.struct([('a', pa.int8()), ('b', pa.int8())]))], pa.struct([('a', pa.int8()), ('b', pa.int8())])), ([(1, 'bar')], [pa.scalar([('a', 1), ('b', 'bar')], type=pa.struct([('a', pa.int8()), ('b', pa.string())]))], pa.struct([('a', pa.int8()), ('b', pa.string())]))])
def test_array_accepts_pyarrow_scalar(seq, data, scalar_data, value_type):
    if type(seq(scalar_data)) == set:
        pytest.skip('The elements in the set get reordered.')
    expect = pa.array(data, type=value_type)
    result = pa.array(seq(scalar_data))
    assert expect.equals(result)
    result = pa.array(seq(scalar_data), type=value_type)
    assert expect.equals(result)