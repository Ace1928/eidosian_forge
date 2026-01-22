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
@pytest.mark.parametrize('indexer', [np.array([1]), [1]])
def test_single_item_array(self, indexer):
    a_del_int = delete(self.a, 1)
    a_del = delete(self.a, indexer)
    assert_equal(a_del_int, a_del)
    nd_a_del_int = delete(self.nd_a, 1, axis=1)
    nd_a_del = delete(self.nd_a, np.array([1]), axis=1)
    assert_equal(nd_a_del_int, nd_a_del)