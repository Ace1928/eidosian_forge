import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, IS_PYPY
def test_dunder_dlpack_stream(self):
    x = np.arange(5)
    x.__dlpack__(stream=None)
    with pytest.raises(RuntimeError):
        x.__dlpack__(stream=1)