import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
@pytest.fixture
def restore_singleton_bitgen():
    """Ensures that the singleton bitgen is restored after a test"""
    orig_bitgen = np.random.get_bit_generator()
    yield
    np.random.set_bit_generator(orig_bitgen)