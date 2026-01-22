import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_hash_agg_random(sum_agg_func_fixture):
    """Test hash aggregate udf with randomly sampled data"""
    value_num = 1000000
    group_num = 1000
    arr1 = pa.array(np.repeat(1, value_num), pa.float64())
    arr2 = pa.array(np.random.choice(group_num, value_num), pa.int32())
    table = pa.table([arr2, arr1], names=['id', 'value'])
    result = table.group_by('id').aggregate([('value', 'sum_udf')])
    expected = table.group_by('id').aggregate([('value', 'sum')]).rename_columns(['id', 'value_sum_udf'])
    assert result.sort_by('id') == expected.sort_by('id')