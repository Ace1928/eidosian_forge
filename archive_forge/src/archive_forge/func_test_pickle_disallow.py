import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_pickle_disallow(tmpdir):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    path = os.path.join(data_dir, 'py2-objarr.npy')
    assert_raises(ValueError, np.load, path, allow_pickle=False, encoding='latin1')
    path = os.path.join(data_dir, 'py2-objarr.npz')
    with np.load(path, allow_pickle=False, encoding='latin1') as f:
        assert_raises(ValueError, f.__getitem__, 'x')
    path = os.path.join(tmpdir, 'pickle-disabled.npy')
    assert_raises(ValueError, np.save, path, np.array([None], dtype=object), allow_pickle=False)