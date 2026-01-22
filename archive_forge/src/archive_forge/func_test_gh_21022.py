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
def test_gh_21022():
    source = np.ma.masked_array(data=[-1, -1], mask=True, dtype=np.float64)
    axis = np.array(0)
    result = np.prod(source, axis=axis, keepdims=False)
    result = np.ma.masked_array(result, mask=np.ones(result.shape, dtype=np.bool_))
    array = np.ma.masked_array(data=-1, mask=True, dtype=np.float64)
    copy.deepcopy(array)
    copy.deepcopy(result)