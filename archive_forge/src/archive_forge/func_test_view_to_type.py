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
def test_view_to_type(self):
    data, a, controlmask = self.data
    test = a.view(np.ndarray)
    assert_(not isinstance(test, MaskedArray))
    assert_equal(test, a._data)
    assert_equal_records(test, data.view(a.dtype).squeeze())