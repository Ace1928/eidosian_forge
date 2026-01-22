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
def test_0d_unicode(self):
    u = 'café'
    utype = type(u)
    arr_nomask = np.ma.array(u)
    arr_masked = np.ma.array(u, mask=True)
    assert_equal(utype(arr_nomask), u)
    assert_equal(utype(arr_masked), '--')