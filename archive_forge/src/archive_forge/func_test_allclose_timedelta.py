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
def test_allclose_timedelta(self):
    a = np.array([[1, 2, 3, 4]], dtype='m8[ns]')
    assert allclose(a, a, atol=0)
    assert allclose(a, a, atol=np.timedelta64(1, 'ns'))