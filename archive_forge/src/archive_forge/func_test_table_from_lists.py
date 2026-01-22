from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_from_lists():
    data = [list(range(5)), [-10, -5, 0, 5, 10]]
    result = pa.table(data, names=['a', 'b'])
    expected = pa.Table.from_arrays(data, names=['a', 'b'])
    assert result.equals(expected)
    schema = pa.schema([pa.field('a', pa.uint16()), pa.field('b', pa.int64())])
    result = pa.table(data, schema=schema)
    expected = pa.Table.from_arrays(data, schema=schema)
    assert result.equals(expected)