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
def test_correct_quantile_value(self):
    a = np.array([True])
    tf_quant = np.quantile(True, False)
    assert_equal(tf_quant, a[0])
    assert_equal(type(tf_quant), a.dtype)
    a = np.array([False, True, True])
    quant_res = np.quantile(a, a)
    assert_array_equal(quant_res, a)
    assert_equal(quant_res.dtype, a.dtype)