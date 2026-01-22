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
@pytest.mark.parametrize('op', [operator.le, operator.lt, operator.ge, operator.gt])
@pytest.mark.parametrize('fill', [None, 'N/A'])
def test_comparisons_strings(self, op, fill):
    ma1 = masked_array(['a', 'b', 'cde'], mask=[0, 1, 0], fill_value=fill)
    ma2 = masked_array(['cde', 'b', 'a'], mask=[0, 1, 0], fill_value=fill)
    assert_equal(op(ma1, ma2)._data, op(ma1._data, ma2._data))