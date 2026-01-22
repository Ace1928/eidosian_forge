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
def test_sign_bit(self):
    x = np.array([0, -0.0, 0])
    assert_equal(str(np.abs(x)), '[0. 0. 0.]')