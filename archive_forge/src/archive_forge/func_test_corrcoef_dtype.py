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
@pytest.mark.parametrize('test_type', [np.half, np.single, np.double, np.longdouble])
def test_corrcoef_dtype(self, test_type):
    cast_A = self.A.astype(test_type)
    res = corrcoef(cast_A, dtype=test_type)
    assert test_type == res.dtype