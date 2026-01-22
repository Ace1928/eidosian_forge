from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_add_column():
    data = [pa.array(range(5)), pa.array([-10, -5, 0, 5, 10]), pa.array(range(5, 10))]
    table = pa.Table.from_arrays(data, names=('a', 'b', 'c'))
    new_field = pa.field('d', data[1].type)
    t2 = table.add_column(3, new_field, data[1])
    t3 = table.append_column(new_field, data[1])
    expected = pa.Table.from_arrays(data + [data[1]], names=('a', 'b', 'c', 'd'))
    assert t2.equals(expected)
    assert t3.equals(expected)
    t4 = table.add_column(0, new_field, data[1])
    expected = pa.Table.from_arrays([data[1]] + data, names=('d', 'a', 'b', 'c'))
    assert t4.equals(expected)