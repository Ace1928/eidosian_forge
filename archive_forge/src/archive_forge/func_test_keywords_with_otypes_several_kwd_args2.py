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
def test_keywords_with_otypes_several_kwd_args2(self):
    f = vectorize(_foo2, otypes=[float])
    r1 = f(z=100, x=10.4, y=-1)
    r2 = f(1, 2, 3)
    assert_equal(r1, _foo2(z=100, x=10.4, y=-1))
    assert_equal(r2, _foo2(1, 2, 3))