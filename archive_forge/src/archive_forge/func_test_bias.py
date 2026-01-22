import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
def test_bias(self):
    with suppress_warnings() as sup:
        warnings.simplefilter('always')
        assert_warns(DeprecationWarning, corrcoef, self.A, self.B, 1, 0)
        assert_warns(DeprecationWarning, corrcoef, self.A, bias=0)
        sup.filter(DeprecationWarning)
        assert_almost_equal(corrcoef(self.A, bias=1), self.res1)