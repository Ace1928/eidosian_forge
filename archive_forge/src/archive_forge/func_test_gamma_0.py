import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_gamma_0(self):
    assert_equal(random.gamma(shape=0, scale=0), 0)
    assert_raises(ValueError, random.gamma, shape=-0.0, scale=-0.0)