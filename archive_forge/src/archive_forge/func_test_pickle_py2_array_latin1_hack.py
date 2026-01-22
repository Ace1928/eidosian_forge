import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_pickle_py2_array_latin1_hack(self):
    data = b"cnumpy.core.multiarray\n_reconstruct\np0\n(cnumpy\nndarray\np1\n(I0\ntp2\nS'b'\np3\ntp4\nRp5\n(I1\n(I1\ntp6\ncnumpy\ndtype\np7\n(S'i1'\np8\nI0\nI1\ntp9\nRp10\n(I3\nS'|'\np11\nNNNI-1\nI-1\nI0\ntp12\nbI00\nS'\\x81'\np13\ntp14\nb."
    result = pickle.loads(data, encoding='latin1')
    assert_array_equal(result, np.array([129]).astype('b'))
    assert_raises(Exception, pickle.loads, data, encoding='koi8-r')