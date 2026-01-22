from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.acero
def test_table_group_by_first():
    table1 = pa.table({'a': [1, 2, 3, 4], 'b': ['a', 'b'] * 2})
    table2 = pa.table({'a': [1, 2, 3, 4], 'b': ['b', 'a'] * 2})
    table = pa.concat_tables([table1, table2])
    with pytest.raises(NotImplementedError):
        table.group_by('b').aggregate([('a', 'first')])
    result = table.group_by('b', use_threads=False).aggregate([('a', 'first')])
    expected = pa.table({'b': ['a', 'b'], 'a_first': [1, 2]})
    assert result.equals(expected)