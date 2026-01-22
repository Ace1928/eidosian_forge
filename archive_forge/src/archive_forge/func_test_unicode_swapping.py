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
def test_unicode_swapping(self):
    ulen = 1
    ucs_value = '\U0010ffff'
    ua = np.array([[[ucs_value * ulen] * 2] * 3] * 4, dtype='U%s' % ulen)
    ua.newbyteorder()