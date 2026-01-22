from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_rename_columns():
    data = [pa.array(range(5)), pa.array([-10, -5, 0, 5, 10]), pa.array(range(5, 10))]
    table = pa.Table.from_arrays(data, names=['a', 'b', 'c'])
    assert table.column_names == ['a', 'b', 'c']
    t2 = table.rename_columns(['eh', 'bee', 'sea'])
    t2.validate()
    assert t2.column_names == ['eh', 'bee', 'sea']
    expected = pa.Table.from_arrays(data, names=['eh', 'bee', 'sea'])
    assert t2.equals(expected)