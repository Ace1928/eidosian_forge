import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
def test_direct_read_dictionary_subfield():
    repeats = 10
    nunique = 5
    data = [[[util.rands(10)] for i in range(nunique)] * repeats]
    table = pa.table(data, names=['f0'])
    bio = pa.BufferOutputStream()
    pq.write_table(table, bio)
    contents = bio.getvalue()
    result = pq.read_table(pa.BufferReader(contents), read_dictionary=['f0.list.element'])
    arr = pa.array(data[0])
    values_as_dict = arr.values.dictionary_encode()
    inner_indices = values_as_dict.indices.cast('int32')
    new_values = pa.DictionaryArray.from_arrays(inner_indices, values_as_dict.dictionary)
    offsets = pa.array(range(51), type='int32')
    expected_arr = pa.ListArray.from_arrays(offsets, new_values)
    expected = pa.table([expected_arr], names=['f0'])
    assert result.equals(expected)
    assert result[0].num_chunks == 1