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
def test_all_zero(self):
    for _arr in self.values():
        arr = np.zeros_like(_arr, dtype=_arr.dtype)
        res1 = trim_zeros(arr, trim='B')
        assert len(res1) == 0
        res2 = trim_zeros(arr, trim='f')
        assert len(res2) == 0