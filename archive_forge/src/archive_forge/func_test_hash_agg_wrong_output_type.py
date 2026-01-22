import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_hash_agg_wrong_output_type(wrong_output_type_agg_func_fixture):
    arr1 = pa.array([10, 20, 30, 40, 50], pa.int64())
    arr2 = pa.array([4, 2, 1, 2, 1], pa.int32())
    table = pa.table([arr2, arr1], names=['id', 'value'])
    with pytest.raises(pa.ArrowTypeError, match='output type'):
        table.group_by('id').aggregate([('value', 'y=wrong_output_type(x)')])