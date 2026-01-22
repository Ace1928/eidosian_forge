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
def test_drop_null_table():
    table = pa.table([pa.array(['a', None, 'c', 'd', None])], names=['a'])
    expected = pa.table([pa.array(['a', 'c', 'd'])], names=['a'])
    result = table.drop_null()
    assert result.equals(expected)
    table = pa.table([pa.chunked_array([['a', None], ['c', 'd', None]]), pa.chunked_array([['a', None], [None, 'd', None]]), pa.chunked_array([['a'], ['b'], [None], ['d', None]])], names=['a', 'b', 'c'])
    expected = pa.table([pa.array(['a', 'd']), pa.array(['a', 'd']), pa.array(['a', 'd'])], names=['a', 'b', 'c'])
    result = table.drop_null()
    assert result.equals(expected)
    table = pa.table([pa.chunked_array([['a', 'b'], ['c', 'd', 'e']]), pa.chunked_array([['A'], ['B'], [None], ['D', None]]), pa.chunked_array([['a`', None], ['c`', 'd`', None]])], names=['a', 'b', 'c'])
    expected = pa.table([pa.array(['a', 'd']), pa.array(['A', 'D']), pa.array(['a`', 'd`'])], names=['a', 'b', 'c'])
    result = table.drop_null()
    assert result.equals(expected)