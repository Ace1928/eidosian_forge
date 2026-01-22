from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_concat_tables():
    data = [list(range(5)), [-10.0, -5.0, 0.0, 5.0, 10.0]]
    data2 = [list(range(5, 10)), [1.0, 2.0, 3.0, 4.0, 5.0]]
    t1 = pa.Table.from_arrays([pa.array(x) for x in data], names=('a', 'b'))
    t2 = pa.Table.from_arrays([pa.array(x) for x in data2], names=('a', 'b'))
    result = pa.concat_tables([t1, t2])
    result.validate()
    assert len(result) == 10
    expected = pa.Table.from_arrays([pa.array(x + y) for x, y in zip(data, data2)], names=('a', 'b'))
    assert result.equals(expected)