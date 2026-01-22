from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_concat_tables_with_promote_option():
    t1 = pa.Table.from_arrays([pa.array([1, 2], type=pa.int64())], ['int64_field'])
    t2 = pa.Table.from_arrays([pa.array([1.0, 2.0], type=pa.float32())], ['float_field'])
    with pytest.warns(FutureWarning):
        result = pa.concat_tables([t1, t2], promote=True)
    assert result.equals(pa.Table.from_arrays([pa.array([1, 2, None, None], type=pa.int64()), pa.array([None, None, 1.0, 2.0], type=pa.float32())], ['int64_field', 'float_field']))
    t1 = pa.Table.from_arrays([pa.array([1, 2], type=pa.int64())], ['f'])
    t2 = pa.Table.from_arrays([pa.array([1, 2], type=pa.float32())], ['f'])
    with pytest.raises(pa.ArrowInvalid, match='Schema at index 1 was different:'):
        with pytest.warns(FutureWarning):
            pa.concat_tables([t1, t2], promote=False)