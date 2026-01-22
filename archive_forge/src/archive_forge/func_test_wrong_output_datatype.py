import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_wrong_output_datatype(wrong_output_datatype_func_fixture):
    _, func_name = wrong_output_datatype_func_fixture
    expected_expr = 'Expected output datatype int16, but function returned datatype int64'
    with pytest.raises(TypeError, match=expected_expr):
        pc.call_function(func_name, [pa.array([20, 30])])