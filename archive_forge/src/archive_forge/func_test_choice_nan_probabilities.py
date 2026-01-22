import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_choice_nan_probabilities(self):
    a = np.array([42, 1, 2])
    p = [None, None, None]
    assert_raises(ValueError, random.choice, a, p=p)