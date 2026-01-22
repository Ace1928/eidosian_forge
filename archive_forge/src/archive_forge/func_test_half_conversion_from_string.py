import platform
import pytest
import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM
@pytest.mark.parametrize('string_dt', ['S', 'U'])
def test_half_conversion_from_string(self, string_dt):
    string = np.array('3.1416', dtype=string_dt)
    assert string.astype(np.float16) == np.array(3.1416, dtype=np.float16)