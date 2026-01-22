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
def test_mvoid_multidim_print(self):
    t_ma = masked_array(data=[([1, 2, 3],)], mask=[([False, True, False],)], fill_value=([999999, 999999, 999999],), dtype=[('a', '<i4', (3,))])
    assert_(str(t_ma[0]) == '([1, --, 3],)')
    assert_(repr(t_ma[0]) == '([1, --, 3],)')
    t_2d = masked_array(data=[([[1, 2], [3, 4]],)], mask=[([[False, True], [True, False]],)], dtype=[('a', '<i4', (2, 2))])
    assert_(str(t_2d[0]) == '([[1, --], [--, 4]],)')
    assert_(repr(t_2d[0]) == '([[1, --], [--, 4]],)')
    t_0d = masked_array(data=[(1, 2)], mask=[(True, False)], dtype=[('a', '<i4'), ('b', '<i4')])
    assert_(str(t_0d[0]) == '(--, 2)')
    assert_(repr(t_0d[0]) == '(--, 2)')
    t_2d = masked_array(data=[([[1, 2], [3, 4]], 1)], mask=[([[False, True], [True, False]], False)], dtype=[('a', '<i4', (2, 2)), ('b', float)])
    assert_(str(t_2d[0]) == '([[1, --], [--, 4]], 1.0)')
    assert_(repr(t_2d[0]) == '([[1, --], [--, 4]], 1.0)')
    t_ne = masked_array(data=[(1, (1, 1))], mask=[(True, (True, False))], dtype=[('a', '<i4'), ('b', 'i4,i4')])
    assert_(str(t_ne[0]) == '(--, (--, 1))')
    assert_(repr(t_ne[0]) == '(--, (--, 1))')