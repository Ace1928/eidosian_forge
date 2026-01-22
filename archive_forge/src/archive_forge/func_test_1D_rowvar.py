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
def test_1D_rowvar(self):
    assert_allclose(cov(self.x3), cov(self.x3, rowvar=False))
    y = np.array([0.078, 0.3107, 0.2111, 0.0334, 0.8501])
    assert_allclose(cov(self.x3, y), cov(self.x3, y, rowvar=False))