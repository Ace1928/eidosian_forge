import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_dirichlet_alpha_non_contiguous(self):
    a = np.array([51.72840233779265, -1.0, 39.74494232180944])
    alpha = a[::2]
    random.seed(self.seed)
    non_contig = random.dirichlet(alpha, size=(3, 2))
    random.seed(self.seed)
    contig = random.dirichlet(np.ascontiguousarray(alpha), size=(3, 2))
    assert_array_almost_equal(non_contig, contig)