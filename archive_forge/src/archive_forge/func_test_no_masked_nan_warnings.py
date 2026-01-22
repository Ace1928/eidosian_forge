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
def test_no_masked_nan_warnings(self):
    m = np.ma.array([0.5, np.nan], mask=[0, 1])
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        exp(m)
        add(m, 1)
        m > 0
        sqrt(m)
        log(m)
        tan(m)
        arcsin(m)
        arccos(m)
        arccosh(m)
        divide(m, 2)
        allclose(m, 0.5)