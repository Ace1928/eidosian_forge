import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_hash_agg_empty(unary_agg_func_fixture):
    arr1 = pa.array([], pa.float64())
    arr2 = pa.array([], pa.int32())
    table = pa.table([arr2, arr1], names=['id', 'value'])
    result = table.group_by('id').aggregate([('value', 'mean_udf')])
    expected = pa.table([pa.array([], pa.int32()), pa.array([], pa.float64())], names=['id', 'value_mean_udf'])
    assert result == expected