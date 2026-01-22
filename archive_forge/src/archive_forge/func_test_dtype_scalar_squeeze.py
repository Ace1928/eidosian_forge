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
def test_dtype_scalar_squeeze(self):
    values = {'S': b'a', 'M': '2018-06-20'}
    for ch in np.typecodes['All']:
        if ch in 'O':
            continue
        sctype = np.dtype(ch).type
        scvalue = sctype(values.get(ch, 3))
        for axis in [None, ()]:
            squeezed = scvalue.squeeze(axis=axis)
            assert_equal(squeezed, scvalue)
            assert_equal(type(squeezed), type(scvalue))