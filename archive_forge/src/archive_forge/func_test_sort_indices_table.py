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
def test_sort_indices_table():
    table = pa.table({'a': [1, 1, None, 0], 'b': [1, 0, 0, 1]})
    result = pc.sort_indices(table, sort_keys=[('a', 'ascending')])
    assert result.to_pylist() == [3, 0, 1, 2]
    result = pc.sort_indices(table, sort_keys=[(pc.field('a'), 'ascending')], null_placement='at_start')
    assert result.to_pylist() == [2, 3, 0, 1]
    result = pc.sort_indices(table, sort_keys=[('a', 'descending'), ('b', 'ascending')])
    assert result.to_pylist() == [1, 0, 3, 2]
    result = pc.sort_indices(table, sort_keys=[('a', 'descending'), ('b', 'ascending')], null_placement='at_start')
    assert result.to_pylist() == [2, 1, 0, 3]
    result = pc.sort_indices(table, [('a', 'descending'), ('b', 'ascending')], null_placement='at_start')
    assert result.to_pylist() == [2, 1, 0, 3]
    with pytest.raises(ValueError, match='Must specify one or more sort keys'):
        pc.sort_indices(table)
    with pytest.raises(ValueError, match='Invalid sort key column: No match for.*unknown'):
        pc.sort_indices(table, sort_keys=[('unknown', 'ascending')])
    with pytest.raises(ValueError, match='not a valid sort order'):
        pc.sort_indices(table, sort_keys=[('a', 'nonscending')])