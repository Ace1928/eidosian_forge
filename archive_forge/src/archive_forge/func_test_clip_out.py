import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def test_clip_out(self):
    a = np.arange(10)
    m = np.ma.MaskedArray(a, mask=[0, 1] * 5)
    m.clip(0, 5, out=m)
    assert_equal(m.mask, [0, 1] * 5)