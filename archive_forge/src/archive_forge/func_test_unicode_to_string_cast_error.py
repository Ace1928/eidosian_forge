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
def test_unicode_to_string_cast_error(self):
    a = np.array(['\x80'] * 129, dtype='U3')
    assert_raises(UnicodeEncodeError, np.array, a, 'S')
    b = a.reshape(3, 43)[:-1, :-1]
    assert_raises(UnicodeEncodeError, np.array, b, 'S')