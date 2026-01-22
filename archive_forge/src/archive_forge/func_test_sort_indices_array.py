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
def test_sort_indices_array():
    arr = pa.array([1, 2, None, 0])
    result = pc.sort_indices(arr)
    assert result.to_pylist() == [3, 0, 1, 2]
    result = pc.sort_indices(arr, sort_keys=[('dummy', 'ascending')])
    assert result.to_pylist() == [3, 0, 1, 2]
    result = pc.sort_indices(arr, sort_keys=[('dummy', 'descending')])
    assert result.to_pylist() == [1, 0, 3, 2]
    result = pc.sort_indices(arr, sort_keys=[('dummy', 'descending')], null_placement='at_start')
    assert result.to_pylist() == [2, 1, 0, 3]
    result = pc.sort_indices(arr, [('dummy', 'descending')], null_placement='at_start')
    assert result.to_pylist() == [2, 1, 0, 3]
    result = pc.sort_indices(arr, options=pc.SortOptions(sort_keys=[('dummy', 'descending')]))
    assert result.to_pylist() == [1, 0, 3, 2]
    result = pc.sort_indices(arr, options=pc.SortOptions(sort_keys=[('dummy', 'descending')], null_placement='at_start'))
    assert result.to_pylist() == [2, 1, 0, 3]