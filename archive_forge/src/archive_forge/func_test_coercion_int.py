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
def test_coercion_int(self):
    a_i = np.zeros((), int)
    assert_raises(MaskError, operator.setitem, a_i, (), np.ma.masked)
    assert_raises(MaskError, int, np.ma.masked)