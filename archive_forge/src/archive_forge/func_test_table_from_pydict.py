from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.parametrize('cls', [pa.Table, pa.RecordBatch])
def test_table_from_pydict(cls):
    table = cls.from_pydict({})
    assert table.num_columns == 0
    assert table.num_rows == 0
    assert table.schema == pa.schema([])
    assert table.to_pydict() == {}
    schema = pa.schema([('strs', pa.utf8()), ('floats', pa.float64())])
    data = OrderedDict([('strs', ['', 'foo', 'bar']), ('floats', [4.5, 5, None])])
    table = cls.from_pydict(data)
    assert table.num_columns == 2
    assert table.num_rows == 3
    assert table.schema == schema
    assert table.to_pydict() == data
    metadata = {b'foo': b'bar'}
    schema = schema.with_metadata(metadata)
    table = cls.from_pydict(data, metadata=metadata)
    assert table.schema == schema
    assert table.schema.metadata == metadata
    assert table.to_pydict() == data
    table = cls.from_pydict(data, schema=schema)
    assert table.schema == schema
    assert table.schema.metadata == metadata
    assert table.to_pydict() == data
    with pytest.raises(ValueError):
        cls.from_pydict(data, schema=schema, metadata=metadata)
    with pytest.raises(TypeError):
        cls.from_pydict({'c0': [0, 1, 2]}, schema=pa.schema([('c0', pa.string())]))
    with pytest.raises(KeyError, match="doesn't contain.* c, d"):
        cls.from_pydict({'a': [1, 2, 3], 'b': [3, 4, 5]}, schema=pa.schema([('a', pa.int64()), ('c', pa.int32()), ('d', pa.int16())]))
    with pytest.raises(TypeError):
        cls.from_pydict({'a': [1, 2, 3]}, schema={})