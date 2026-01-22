from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_unify_dictionaries():
    batch1 = pa.record_batch([pa.array(['foo', 'bar', None, 'foo']).dictionary_encode(), pa.array([123, 456, 456, 789]).dictionary_encode(), pa.array([True, False, None, None])], names=['a', 'b', 'c'])
    batch2 = pa.record_batch([pa.array(['quux', 'foo', None, 'quux']).dictionary_encode(), pa.array([456, 789, 789, None]).dictionary_encode(), pa.array([False, None, None, True])], names=['a', 'b', 'c'])
    table = pa.Table.from_batches([batch1, batch2])
    table = table.replace_schema_metadata({b'key1': b'value1'})
    assert table.column(0).chunk(0).dictionary.equals(pa.array(['foo', 'bar']))
    assert table.column(0).chunk(1).dictionary.equals(pa.array(['quux', 'foo']))
    assert table.column(1).chunk(0).dictionary.equals(pa.array([123, 456, 789]))
    assert table.column(1).chunk(1).dictionary.equals(pa.array([456, 789]))
    table = table.unify_dictionaries(pa.default_memory_pool())
    expected_dict_0 = pa.array(['foo', 'bar', 'quux'])
    expected_dict_1 = pa.array([123, 456, 789])
    assert table.column(0).chunk(0).dictionary.equals(expected_dict_0)
    assert table.column(0).chunk(1).dictionary.equals(expected_dict_0)
    assert table.column(1).chunk(0).dictionary.equals(expected_dict_1)
    assert table.column(1).chunk(1).dictionary.equals(expected_dict_1)
    assert table.to_pydict() == {'a': ['foo', 'bar', None, 'foo', 'quux', 'foo', None, 'quux'], 'b': [123, 456, 456, 789, 456, 789, 789, None], 'c': [True, False, None, None, False, None, None, True]}
    assert table.schema.metadata == {b'key1': b'value1'}