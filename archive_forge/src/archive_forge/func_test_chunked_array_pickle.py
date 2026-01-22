from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.parametrize(('data', 'typ'), [([True, False, True, True], pa.bool_()), ([1, 2, 4, 6], pa.int64()), ([1.0, 2.5, None], pa.float64()), (['a', None, 'b'], pa.string()), ([], pa.list_(pa.uint8())), ([[1, 2], [3]], pa.list_(pa.int64())), ([['a'], None, ['b', 'c']], pa.list_(pa.string())), ([(1, 'a'), (2, 'c'), None], pa.struct([pa.field('a', pa.int64()), pa.field('b', pa.string())]))])
def test_chunked_array_pickle(data, typ, pickle_module):
    arrays = []
    while data:
        arrays.append(pa.array(data[:2], type=typ))
        data = data[2:]
    array = pa.chunked_array(arrays, type=typ)
    array.validate()
    result = pickle_module.loads(pickle_module.dumps(array))
    result.validate()
    assert result.equals(array)