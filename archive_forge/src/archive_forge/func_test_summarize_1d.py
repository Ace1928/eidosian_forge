import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_summarize_1d(self):
    A = np.arange(1001)
    strA = '[   0    1    2 ...  998  999 1000]'
    assert_equal(str(A), strA)
    reprA = 'array([   0,    1,    2, ...,  998,  999, 1000])'
    assert_equal(repr(A), reprA)