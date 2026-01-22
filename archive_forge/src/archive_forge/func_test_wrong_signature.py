import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_wrong_signature(wrong_signature_func_fixture):
    _, func_name = wrong_signature_func_fixture
    expected_expr = 'wrong_signature\\(\\) takes 0 positional arguments but 1 was given'
    with pytest.raises(TypeError, match=expected_expr):
        pc.call_function(func_name, [], length=1)