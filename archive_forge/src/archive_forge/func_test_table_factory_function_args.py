from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_factory_function_args():
    with pytest.raises(ValueError):
        pa.table({'a': [1, 2, 3]}, names=['a'])
    schema = pa.schema([('a', pa.int32())])
    table = pa.table({'a': pa.array([1, 2, 3], type=pa.int64())}, schema)
    assert table.column('a').type == pa.int32()
    data = [pa.array([1, 2, 3], type='int64')]
    names = ['a']
    table = pa.table(data, names)
    assert table.column_names == names
    schema = pa.schema([('a', pa.int64())])
    table = pa.table(data, schema)
    assert table.column_names == names