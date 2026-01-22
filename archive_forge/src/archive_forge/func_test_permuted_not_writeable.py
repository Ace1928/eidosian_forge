import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_permuted_not_writeable(self):
    x = np.zeros((2, 5))
    x.flags.writeable = False
    with pytest.raises(ValueError, match='read-only'):
        random.permuted(x, axis=1, out=x)