import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_invalid_prob(self):
    assert_raises(ValueError, random.multinomial, 100, [1.1, 0.2])
    assert_raises(ValueError, random.multinomial, 100, [-0.1, 0.9])