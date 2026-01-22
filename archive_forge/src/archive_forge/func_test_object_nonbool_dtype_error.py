import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_object_nonbool_dtype_error(self):
    assert np.equal(1, [1], dtype=bool).dtype == bool
    with pytest.raises(TypeError, match='No loop matching'):
        np.equal(1, 1, dtype=np.int64)
    with pytest.raises(TypeError, match='No loop matching'):
        np.equal(1, 1, sig=(None, None, 'l'))