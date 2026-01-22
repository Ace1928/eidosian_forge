import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('bound, expected', [(2 ** 32 - 1, np.array([517043486, 1364798665, 1733884389, 1353720612, 3769704066, 1170797179, 4108474671])), (2 ** 32, np.array([517043487, 1364798666, 1733884390, 1353720613, 3769704067, 1170797180, 4108474672])), (2 ** 32 + 1, np.array([517043487, 1733884390, 3769704068, 4108474673, 1831631863, 1215661561, 3869512430]))])
def test_repeatability_32bit_boundary(self, bound, expected):
    for size in [None, len(expected)]:
        random = Generator(MT19937(1234))
        x = random.integers(bound, size=size)
        assert_equal(x, expected if size is not None else expected[0])