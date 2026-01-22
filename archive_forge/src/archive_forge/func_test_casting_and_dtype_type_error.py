import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_casting_and_dtype_type_error(self):
    a = np.array([1, 2, 3])
    b = np.array([2.5, 3.5, 4.5])
    with pytest.raises(TypeError):
        vstack((a, b), casting='safe', dtype=np.int64)