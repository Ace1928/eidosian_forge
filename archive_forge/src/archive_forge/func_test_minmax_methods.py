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
def test_minmax_methods(self):
    _, _, _, _, _, xm, _, _, _, _ = self.d
    xm.shape = (xm.size,)
    assert_equal(xm.max(), 10)
    assert_(xm[0].max() is masked)
    assert_(xm[0].max(0) is masked)
    assert_(xm[0].max(-1) is masked)
    assert_equal(xm.min(), -10.0)
    assert_(xm[0].min() is masked)
    assert_(xm[0].min(0) is masked)
    assert_(xm[0].min(-1) is masked)
    assert_equal(xm.ptp(), 20.0)
    assert_(xm[0].ptp() is masked)
    assert_(xm[0].ptp(0) is masked)
    assert_(xm[0].ptp(-1) is masked)
    x = array([1, 2, 3], mask=True)
    assert_(x.min() is masked)
    assert_(x.max() is masked)
    assert_(x.ptp() is masked)