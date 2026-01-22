from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_select():
    a1 = pa.array([1, 2, 3, None, 5])
    a2 = pa.array(['a', 'b', 'c', 'd', 'e'])
    a3 = pa.array([[1, 2], [3, 4], [5, 6], None, [9, 10]])
    batch = pa.record_batch([a1, a2, a3], ['f1', 'f2', 'f3'])
    result = batch.select(['f1'])
    expected = pa.record_batch([a1], ['f1'])
    assert result.equals(expected)
    result = batch.select(['f3', 'f2'])
    expected = pa.record_batch([a3, a2], ['f3', 'f2'])
    assert result.equals(expected)
    result = batch.select([0])
    expected = pa.record_batch([a1], ['f1'])
    assert result.equals(expected)
    result = batch.select([2, 1])
    expected = pa.record_batch([a3, a2], ['f3', 'f2'])
    assert result.equals(expected)
    batch2 = batch.replace_schema_metadata({'a': 'test'})
    result = batch2.select(['f1', 'f2'])
    assert b'a' in result.schema.metadata
    with pytest.raises(KeyError, match='Field "f5" does not exist'):
        batch.select(['f5'])
    with pytest.raises(IndexError, match='index out of bounds'):
        batch.select([5])
    result = batch.select(['f2', 'f2'])
    expected = pa.record_batch([a2, a2], ['f2', 'f2'])
    assert result.equals(expected)
    batch = pa.record_batch([a1, a2, a3], ['f1', 'f2', 'f1'])
    with pytest.raises(KeyError, match='Field "f1" exists 2 times'):
        batch.select(['f1'])
    result = batch.select(['f2'])
    expected = pa.record_batch([a2], ['f2'])
    assert result.equals(expected)