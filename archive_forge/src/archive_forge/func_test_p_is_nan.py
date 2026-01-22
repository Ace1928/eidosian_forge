import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_p_is_nan(self):
    assert_raises(ValueError, random.binomial, 1, np.nan)