import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_dirichlet_bad_alpha(self):
    alpha = np.array([0.54, -1e-16])
    assert_raises(ValueError, random.dirichlet, alpha)