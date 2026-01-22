import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
@pytest.mark.parametrize('dtypes, expected', [((np.float16, np.float64), np.float64), ((np.float32, np.uint16, np.int8), np.float64), ((np.float32, np.float16), np.float32)])
def test_supported_float_dtype_sequence(dtypes, expected):
    float_dtype = _supported_float_type(dtypes)
    assert float_dtype == expected