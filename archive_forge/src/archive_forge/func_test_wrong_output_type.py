import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_wrong_output_type(wrong_output_type_func_fixture):
    _, func_name = wrong_output_type_func_fixture
    with pytest.raises(TypeError, match='Unexpected output type: int'):
        pc.call_function(func_name, [], length=1)