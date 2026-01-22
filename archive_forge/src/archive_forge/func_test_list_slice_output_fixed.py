from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
@pytest.mark.parametrize('start,stop,expected', ((0, None, [[1, 2, 3], [4, 5, None], [6, None, None], None]), (0, 1, [[1], [4], [6], None]), (0, 2, [[1, 2], [4, 5], [6, None], None]), (1, 2, [[2], [5], [None], None]), (2, 4, [[3, None], [None, None], [None, None], None])))
@pytest.mark.parametrize('step', (1, 2))
@pytest.mark.parametrize('value_type', (pa.string, pa.int16, pa.float64))
@pytest.mark.parametrize('list_type', (pa.list_, pa.large_list, 'fixed'))
def test_list_slice_output_fixed(start, stop, step, expected, value_type, list_type):
    if list_type == 'fixed':
        arr = pa.array([[1, 2, 3], [4, 5, None], [6, None, None], None], pa.list_(pa.int8(), 3)).cast(pa.list_(value_type(), 3))
    else:
        arr = pa.array([[1, 2, 3], [4, 5], [6], None], pa.list_(pa.int8())).cast(list_type(value_type()))
    args = (arr, start, stop, step, True)
    if stop is None and list_type != 'fixed':
        msg = 'Unable to produce FixedSizeListArray from non-FixedSizeListArray without `stop` being set.'
        with pytest.raises(pa.ArrowNotImplementedError, match=msg):
            pc.list_slice(*args)
    else:
        result = pc.list_slice(*args)
        pylist = result.cast(pa.list_(pa.int8(), result.type.list_size)).to_pylist()
        assert pylist == [e[::step] if e else e for e in expected]