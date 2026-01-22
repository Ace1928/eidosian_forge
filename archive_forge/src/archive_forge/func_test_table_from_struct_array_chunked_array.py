from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_from_struct_array_chunked_array():
    chunked_struct_array = pa.chunked_array([[{'ints': 1}, {'floats': 1.0}]], type=pa.struct([('ints', pa.int32()), ('floats', pa.float32())]))
    result = pa.Table.from_struct_array(chunked_struct_array)
    assert result.equals(pa.Table.from_arrays([pa.array([1, None], type=pa.int32()), pa.array([None, 1.0], type=pa.float32())], ['ints', 'floats']))