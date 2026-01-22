from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_select_column():
    data = [pa.array(range(5)), pa.array([-10, -5, 0, 5, 10]), pa.array(range(5, 10))]
    table = pa.Table.from_arrays(data, names=('a', 'b', 'c'))
    assert table.column('a').equals(table.column(0))
    with pytest.raises(KeyError, match='Field "d" does not exist in schema'):
        table.column('d')
    with pytest.raises(TypeError):
        table.column(None)
    with pytest.raises(IndexError):
        table.column(4)