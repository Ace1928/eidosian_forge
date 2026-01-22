import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_gaussian_reset(self):
    old = self.random_state.standard_normal(size=3)
    self.random_state.set_state(self.state)
    new = self.random_state.standard_normal(size=3)
    assert_(np.all(old == new))