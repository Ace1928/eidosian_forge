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
def test_mvoid_print(self):
    mx = array([(1, 1), (2, 2)], dtype=[('a', int), ('b', int)])
    assert_equal(str(mx[0]), '(1, 1)')
    mx['b'][0] = masked
    ini_display = masked_print_option._display
    masked_print_option.set_display('-X-')
    try:
        assert_equal(str(mx[0]), '(1, -X-)')
        assert_equal(repr(mx[0]), '(1, -X-)')
    finally:
        masked_print_option.set_display(ini_display)
    mx = array([(1,), (2,)], dtype=[('a', 'O')])
    assert_equal(str(mx[0]), '(1,)')