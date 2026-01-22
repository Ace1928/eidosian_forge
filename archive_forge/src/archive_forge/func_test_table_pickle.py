from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_pickle(pickle_module):
    data = [pa.chunked_array([[1, 2], [3, 4]], type=pa.uint32()), pa.chunked_array([['some', 'strings', None, '']], type=pa.string())]
    schema = pa.schema([pa.field('ints', pa.uint32()), pa.field('strs', pa.string())], metadata={b'foo': b'bar'})
    table = pa.Table.from_arrays(data, schema=schema)
    result = pickle_module.loads(pickle_module.dumps(table))
    result.validate()
    assert result.equals(table)