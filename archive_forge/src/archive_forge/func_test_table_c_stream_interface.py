from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_c_stream_interface():

    class StreamWrapper:

        def __init__(self, batches):
            self.batches = batches

        def __arrow_c_stream__(self, requested_schema=None):
            reader = pa.RecordBatchReader.from_batches(self.batches[0].schema, self.batches)
            return reader.__arrow_c_stream__(requested_schema)
    data = [pa.record_batch([pa.array([1, 2, 3], type=pa.int64())], names=['a']), pa.record_batch([pa.array([4, 5, 6], type=pa.int64())], names=['a'])]
    wrapper = StreamWrapper(data)
    result = pa.table(wrapper)
    expected = pa.Table.from_batches(data)
    assert result == expected
    result = pa.table(wrapper, schema=data[0].schema)
    assert result == expected
    with pytest.raises(NotImplementedError):
        pa.table(wrapper, schema=pa.schema([pa.field('a', pa.int32())]))