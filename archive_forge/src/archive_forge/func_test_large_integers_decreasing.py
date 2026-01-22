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
@pytest.mark.xfail(reason='gh-11022: np.core.multiarray._monoticity loses precision')
def test_large_integers_decreasing(self):
    x = 2 ** 54
    assert_equal(np.digitize(x, [x + 1, x - 1]), 1)