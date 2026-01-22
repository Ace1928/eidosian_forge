import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
def test_dictionary_array_automatically_read():
    dict_length = 4000
    dict_values = pa.array(['x' * 1000 + '_{}'.format(i) for i in range(dict_length)])
    num_chunks = 10
    chunk_size = 100
    chunks = []
    for i in range(num_chunks):
        indices = np.random.randint(0, dict_length, size=chunk_size).astype(np.int32)
        chunks.append(pa.DictionaryArray.from_arrays(pa.array(indices), dict_values))
    table = pa.table([pa.chunked_array(chunks)], names=['f0'])
    result = _simple_table_write_read(table)
    assert result.equals(table)
    assert result.schema.metadata is None