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
def test_basic0d(self):
    x = masked_array(0)
    assert_equal(str(x), '0')
    x = masked_array(0, mask=True)
    assert_equal(str(x), str(masked_print_option))
    x = masked_array(0, mask=False)
    assert_equal(str(x), '0')
    x = array(0, mask=1)
    assert_(x.filled().dtype is x._data.dtype)