import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
@pytest.mark.skipif(sys.version_info >= (3, 12), reason='see gh-23988')
@pytest.mark.xfail(IS_WASM, reason='Emscripten NODEFS has a buggy dup')
def test_python2_python3_interoperability():
    fname = 'win64python2.npy'
    path = os.path.join(os.path.dirname(__file__), 'data', fname)
    with pytest.warns(UserWarning, match='Reading.*this warning\\.'):
        data = np.load(path)
    assert_array_equal(data, np.ones(2))