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
def test_complex_scalar_complex_cast(self):
    for tp in [np.csingle, np.cdouble, np.clongdouble]:
        x = tp(1 + 2j)
        assert_equal(complex(x), 1 + 2j)