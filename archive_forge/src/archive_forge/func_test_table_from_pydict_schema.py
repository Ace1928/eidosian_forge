from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.parametrize('data, klass', [((['', 'foo', 'bar'], [4.5, 5, None]), list), ((['', 'foo', 'bar'], [4.5, 5, None]), pa.array), (([[''], ['foo', 'bar']], [[4.5], [5.0, None]]), pa.chunked_array)])
def test_table_from_pydict_schema(data, klass):
    data = OrderedDict([('strs', klass(data[0])), ('floats', klass(data[1]))])
    schema = pa.schema([('strs', pa.utf8()), ('floats', pa.float64()), ('ints', pa.int64())])
    with pytest.raises(KeyError, match='ints'):
        pa.Table.from_pydict(data, schema=schema)
    schema = pa.schema([('strs', pa.utf8())])
    table = pa.Table.from_pydict(data, schema=schema)
    assert table.num_columns == 1
    assert table.schema == schema
    assert table.column_names == ['strs']