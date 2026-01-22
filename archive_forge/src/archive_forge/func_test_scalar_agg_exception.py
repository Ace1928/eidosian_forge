import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_scalar_agg_exception(exception_agg_func_fixture):
    arr = pa.array([10, 20, 30, 40, 50, 60], pa.int64())
    with pytest.raises(RuntimeError, match='Oops'):
        pc.call_function('y=exception_len(x)', [arr])