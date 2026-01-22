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
def test_with_incorrect_minlength(self):
    x = np.array([], dtype=int)
    assert_raises_regex(TypeError, "'str' object cannot be interpreted", lambda: np.bincount(x, minlength='foobar'))
    assert_raises_regex(ValueError, 'must not be negative', lambda: np.bincount(x, minlength=-1))
    x = np.arange(5)
    assert_raises_regex(TypeError, "'str' object cannot be interpreted", lambda: np.bincount(x, minlength='foobar'))
    assert_raises_regex(ValueError, 'must not be negative', lambda: np.bincount(x, minlength=-1))