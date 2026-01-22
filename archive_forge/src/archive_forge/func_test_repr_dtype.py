import numpy as np
import pytest
import pandas as pd
from pandas.core.arrays.floating import (
@pytest.mark.parametrize('dtype, expected', [(Float32Dtype(), 'Float32Dtype()'), (Float64Dtype(), 'Float64Dtype()')])
def test_repr_dtype(dtype, expected):
    assert repr(dtype) == expected