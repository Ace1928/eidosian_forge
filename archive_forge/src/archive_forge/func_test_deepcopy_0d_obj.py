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
def test_deepcopy_0d_obj():
    source = np.ma.array(0, mask=[0], dtype=object)
    deepcopy = copy.deepcopy(source)
    deepcopy[...] = 17
    assert_equal(source, 0)
    assert_equal(deepcopy, 17)