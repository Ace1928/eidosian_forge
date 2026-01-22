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
def test_unaligned_unicode_access(self):
    for i in range(1, 9):
        msg = 'unicode offset: %d chars' % i
        t = np.dtype([('a', 'S%d' % i), ('b', 'U2')])
        x = np.array([(b'a', 'b')], dtype=t)
        assert_equal(str(x), "[(b'a', 'b')]", err_msg=msg)