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
def test_mask_element(self):
    """Check record access"""
    base = self.data['base']
    base[0] = masked
    for n in ('a', 'b', 'c'):
        assert_equal(base[n].mask, [1, 1, 0, 0, 1])
        assert_equal(base[n]._data, base._data[n])