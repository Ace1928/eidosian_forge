import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def test_hamming_unequal_length_with_w():
    u = [0, 0, 1]
    v = [0, 0, 1]
    w = [1, 0, 1, 0]
    msg = "'w' should have the same length as 'u' and 'v'."
    with pytest.raises(ValueError, match=msg):
        whamming(u, v, w)