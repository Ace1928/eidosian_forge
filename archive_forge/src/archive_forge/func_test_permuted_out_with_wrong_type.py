import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_permuted_out_with_wrong_type(self):
    out = np.zeros((3, 5), dtype=np.int32)
    x = np.ones((3, 5))
    with pytest.raises(TypeError, match='Cannot cast'):
        random.permuted(x, axis=1, out=out)