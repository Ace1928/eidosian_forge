import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_writeable_memoryview():
    original = np.array([1, 2, 3])
    for is_broadcast, results in [(False, broadcast_arrays(original)), (True, broadcast_arrays(0, original))]:
        for result in results:
            if is_broadcast:
                assert memoryview(result).readonly
            else:
                assert not memoryview(result).readonly