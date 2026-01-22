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
def test_rank_options():
    arr = pa.array([1.2, 0.0, 5.3, None, 5.3, None, 0.0])
    expected = pa.array([3, 1, 4, 6, 5, 7, 2], type=pa.uint64())
    result = pc.rank(arr)
    assert result.equals(expected)
    result = pc.rank(arr, options=pc.RankOptions())
    assert result.equals(expected)
    result = pc.rank(arr, options=pc.RankOptions(sort_keys=[('b', 'ascending')]))
    assert result.equals(expected)
    result = pc.rank(arr, null_placement='at_start')
    expected_at_start = pa.array([5, 3, 6, 1, 7, 2, 4], type=pa.uint64())
    assert result.equals(expected_at_start)
    result = pc.rank(arr, sort_keys='descending')
    expected_descending = pa.array([3, 4, 1, 6, 2, 7, 5], type=pa.uint64())
    assert result.equals(expected_descending)
    with pytest.raises(ValueError, match='"NonExisting" is not a valid tiebreaker'):
        pc.RankOptions(sort_keys='descending', null_placement='at_end', tiebreaker='NonExisting')