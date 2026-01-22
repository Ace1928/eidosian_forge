import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_abstract_type(self):
    assert_raises(KeyError, np.sctype2char, np.floating)