import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_noncentral_f_nan(self):
    random.seed(self.seed)
    actual = random.noncentral_f(dfnum=5, dfden=2, nonc=np.nan)
    assert np.isnan(actual)