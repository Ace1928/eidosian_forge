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
def test_both(self):
    a = rand(10)
    mask = a > 0.5
    ac = a.copy()
    c = extract(mask, a)
    place(a, mask, 0)
    place(a, mask, c)
    assert_array_equal(a, ac)