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
def test_void0d(self):
    ndtype = [('a', int), ('b', int)]
    a = np.array([(1, 2)], dtype=ndtype)[0]
    f = mvoid(a)
    assert_(isinstance(f, mvoid))
    a = masked_array([(1, 2)], mask=[(1, 0)], dtype=ndtype)[0]
    assert_(isinstance(a, mvoid))
    a = masked_array([(1, 2), (1, 2)], mask=[(1, 0), (0, 0)], dtype=ndtype)
    f = mvoid(a._data[0], a._mask[0])
    assert_(isinstance(f, mvoid))