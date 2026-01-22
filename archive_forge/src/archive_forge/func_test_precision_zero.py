import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_precision_zero(self):
    np.set_printoptions(precision=0)
    for values, string in (([0.0], '0.'), ([0.3], '0.'), ([-0.3], '-0.'), ([0.7], '1.'), ([1.5], '2.'), ([-1.5], '-2.'), ([-15.34], '-15.'), ([100.0], '100.'), ([0.2, -1, 122.51], '  0.,  -1., 123.'), ([0], '0'), ([-12], '-12'), ([complex(0.3, -0.7)], '0.-1.j')):
        x = np.array(values)
        assert_equal(repr(x), 'array([%s])' % string)