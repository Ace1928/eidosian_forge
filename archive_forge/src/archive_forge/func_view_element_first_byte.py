import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def view_element_first_byte(x):
    """Construct an array viewing the first byte of each element of `x`"""
    from numpy.lib.stride_tricks import DummyArray
    interface = dict(x.__array_interface__)
    interface['typestr'] = '|b1'
    interface['descr'] = [('', '|b1')]
    return np.asarray(DummyArray(interface, x))