import os
import pytest
import numpy as np
from . import util
@pytest.mark.slow
def test_constant_integer_long(self):
    x = np.arange(6, dtype=np.int64)[::2]
    pytest.raises(ValueError, self.module.foo_long, x)
    x = np.arange(3, dtype=np.int64)
    self.module.foo_long(x)
    assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])