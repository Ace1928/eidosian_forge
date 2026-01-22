import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_generic_rank1(self):
    """Test rank 1 array for all dtypes."""

    def foo(t):
        a = np.empty(2, t)
        a.fill(1)
        b = a.copy()
        c = a.copy()
        c.fill(0)
        self._test_equal(a, b)
        self._test_not_equal(c, b)
    for t in '?bhilqpBHILQPfdgFDG':
        foo(t)
    for t in ['S1', 'U1']:
        foo(t)