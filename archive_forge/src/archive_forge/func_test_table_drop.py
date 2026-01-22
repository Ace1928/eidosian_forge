from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_drop():
    """ verify the alias of drop_columns is working"""
    a = pa.array(range(5))
    b = pa.array([-10, -5, 0, 5, 10])
    c = pa.array(range(5, 10))
    table = pa.Table.from_arrays([a, b, c], names=('a', 'b', 'c'))
    t2 = table.drop(['a', 'b'])
    t3 = table.drop('a')
    exp_t2 = pa.Table.from_arrays([c], names=('c',))
    assert exp_t2.equals(t2)
    exp_t3 = pa.Table.from_arrays([b, c], names=('b', 'c'))
    assert exp_t3.equals(t3)
    with pytest.raises(KeyError, match="Column 'd' not found"):
        table.drop(['d'])