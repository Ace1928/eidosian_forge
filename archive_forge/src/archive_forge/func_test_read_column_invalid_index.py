import io
import os
import sys
import pytest
import pyarrow as pa
def test_read_column_invalid_index():
    table = pa.table([pa.array([4, 5]), pa.array(['foo', 'bar'])], names=['ints', 'strs'])
    bio = pa.BufferOutputStream()
    pq.write_table(table, bio)
    f = pq.ParquetFile(bio.getvalue())
    assert f.reader.read_column(0).to_pylist() == [4, 5]
    assert f.reader.read_column(1).to_pylist() == ['foo', 'bar']
    for index in (-1, 2):
        with pytest.raises((ValueError, IndexError)):
            f.reader.read_column(index)