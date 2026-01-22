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
def test_max_ulp(self):
    x = [0.0, 0.2, 0.4]
    a = np.quantile(x, 0.45)
    np.testing.assert_array_max_ulp(a, 0.18, maxulp=1)