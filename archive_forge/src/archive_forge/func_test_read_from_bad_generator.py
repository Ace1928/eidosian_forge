import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_read_from_bad_generator():

    def gen():
        for entry in ['1,2', b'3, 5', 12738]:
            yield entry
    with pytest.raises(TypeError, match='non-string returned while reading data'):
        np.loadtxt(gen(), dtype='i, i', delimiter=',')