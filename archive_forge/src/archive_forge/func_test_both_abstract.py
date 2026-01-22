import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_both_abstract(self):
    assert_(np.issubdtype(np.floating, np.inexact))
    assert_(not np.issubdtype(np.inexact, np.floating))