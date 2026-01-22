from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_c_array_interface():

    class BatchWrapper:

        def __init__(self, batch):
            self.batch = batch

        def __arrow_c_array__(self, requested_schema=None):
            return self.batch.__arrow_c_array__(requested_schema)
    data = pa.record_batch([pa.array([1, 2, 3], type=pa.int64())], names=['a'])
    wrapper = BatchWrapper(data)
    result = pa.record_batch(wrapper)
    assert result == data
    castable_schema = pa.schema([pa.field('a', pa.int32())])
    result = pa.record_batch(wrapper, schema=castable_schema)
    expected = pa.record_batch([pa.array([1, 2, 3], type=pa.int32())], names=['a'])
    assert result == expected