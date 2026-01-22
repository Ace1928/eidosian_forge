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
def test_round_with_scalar(self):
    a = array(1.1, mask=[False])
    assert_equal(a.round(), 1)
    a = array(1.1, mask=[True])
    assert_(a.round() is masked)
    a = array(1.1, mask=[False])
    output = np.empty(1, dtype=float)
    output.fill(-9999)
    a.round(out=output)
    assert_equal(output, 1)
    a = array(1.1, mask=[False])
    output = array(-9999.0, mask=[True])
    a.round(out=output)
    assert_equal(output[()], 1)
    a = array(1.1, mask=[True])
    output = array(-9999.0, mask=[False])
    a.round(out=output)
    assert_(output[()] is masked)