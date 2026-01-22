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
def test_signature_simple(self):

    def addsubtract(a, b):
        if a > b:
            return a - b
        else:
            return a + b
    f = vectorize(addsubtract, signature='(),()->()')
    r = f([0, 3, 6, 9], [1, 3, 5, 7])
    assert_array_equal(r, [1, 6, 1, 2])