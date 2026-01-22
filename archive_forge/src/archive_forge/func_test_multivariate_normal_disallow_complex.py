import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('mean, cov', [([0], [[1 + 1j]]), ([0j], [[1]])])
def test_multivariate_normal_disallow_complex(self, mean, cov):
    random = Generator(MT19937(self.seed))
    with pytest.raises(TypeError, match='must not be complex'):
        random.multivariate_normal(mean, cov)