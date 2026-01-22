import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
@pytest.mark.parametrize('dtype', ['f', 'float32', np.float32, np.dtype(np.float32)])
def test_supported_float_dtype_input_kinds(dtype):
    assert _supported_float_type(dtype) == np.float32