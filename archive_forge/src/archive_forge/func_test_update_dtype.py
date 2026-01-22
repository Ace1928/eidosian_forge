import re
import warnings
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
@pytest.mark.parametrize('original, dtype, expected', [(SparseDtype(int, 0), float, SparseDtype(float, 0.0)), (SparseDtype(int, 1), float, SparseDtype(float, 1.0)), (SparseDtype(int, 1), str, SparseDtype(object, '1')), (SparseDtype(float, 1.5), int, SparseDtype(int, 1))])
def test_update_dtype(original, dtype, expected):
    result = original.update_dtype(dtype)
    assert result == expected