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
def test_percentile_no_overwrite(self):
    a = np.array([2, 3, 4, 1])
    np.percentile(a, [50], overwrite_input=False)
    assert_equal(a, np.array([2, 3, 4, 1]))
    a = np.array([2, 3, 4, 1])
    np.percentile(a, [50])
    assert_equal(a, np.array([2, 3, 4, 1]))