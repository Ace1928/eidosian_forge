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
@pytest.mark.large_memory
@pytest.mark.parametrize(('ty', 'char'), [(pa.string(), 'x'), (pa.binary(), b'x')])
def test_nested_auto_chunking(ty, char):
    v1 = char * 100000000
    v2 = char * 147483646
    struct_type = pa.struct([pa.field('bool', pa.bool_()), pa.field('integer', pa.int64()), pa.field('string-like', ty)])
    data = [{'bool': True, 'integer': 1, 'string-like': v1}] * 20
    data.append({'bool': True, 'integer': 1, 'string-like': v2})
    arr = pa.array(data, type=struct_type)
    assert isinstance(arr, pa.Array)
    data.append({'bool': True, 'integer': 1, 'string-like': char})
    arr = pa.array(data, type=struct_type)
    assert isinstance(arr, pa.ChunkedArray)
    assert arr.num_chunks == 2
    assert len(arr.chunk(0)) == 21
    assert len(arr.chunk(1)) == 1
    assert arr.chunk(1)[0].as_py() == {'bool': True, 'integer': 1, 'string-like': char}