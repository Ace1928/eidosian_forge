from multiprocessing import Pool
from multiprocessing.pool import Pool as PWL
import os
import re
import math
from fractions import Fraction
import numpy as np
from numpy.testing import assert_equal, assert_
import pytest
from pytest import raises as assert_raises, deprecated_call
import scipy
from scipy._lib._util import (_aligned_zeros, check_random_state, MapWrapper,
@pytest.mark.parametrize('key', ('ifft', 'diag', 'arccos', 'randn', 'rand', 'array'))
def test_numpy_deprecation(key):
    """Test that 'from numpy import *' functions are deprecated."""
    if key in ('ifft', 'diag', 'arccos'):
        arg = [1.0, 0.0]
    elif key == 'finfo':
        arg = float
    else:
        arg = 2
    func = getattr(scipy, key)
    match = 'scipy\\.%s is deprecated.*2\\.0\\.0' % key
    with deprecated_call(match=match) as dep:
        func(arg)
    fnames = [os.path.splitext(d.filename)[0] for d in dep.list]
    basenames = [os.path.basename(fname) for fname in fnames]
    assert 'test__util' in basenames
    if key in ('rand', 'randn'):
        root = np.random
    elif key == 'ifft':
        root = np.fft
    else:
        root = np
    func_np = getattr(root, key)
    func_np(arg)
    assert func_np is not func
    if isinstance(func_np, type):
        assert isinstance(func, type)