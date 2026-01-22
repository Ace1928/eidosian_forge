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
def test_signature_invalid_inputs(self):
    f = vectorize(operator.add, signature='(n),(n)->(n)')
    with assert_raises_regex(TypeError, 'wrong number of positional'):
        f([1, 2])
    with assert_raises_regex(ValueError, 'does not have enough dimensions'):
        f(1, 2)
    with assert_raises_regex(ValueError, 'inconsistent size for core dimension'):
        f([1, 2], [1, 2, 3])
    f = vectorize(operator.add, signature='()->()')
    with assert_raises_regex(TypeError, 'wrong number of positional'):
        f(1, 2)