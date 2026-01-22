from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_select_column():
    data = [pa.array(range(5)), pa.array([-10, -5, 0, 5, 10]), pa.array(range(5, 10))]
    batch = pa.RecordBatch.from_arrays(data, names=('a', 'b', 'c'))
    assert batch.column('a').equals(batch.column(0))
    with pytest.raises(KeyError, match='Field "d" does not exist in schema'):
        batch.column('d')
    with pytest.raises(TypeError):
        batch.column(None)
    with pytest.raises(IndexError):
        batch.column(4)