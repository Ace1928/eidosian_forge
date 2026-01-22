import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_ctx_mgr(self):
    with np.printoptions(precision=2):
        s = str(np.array([2.0]) / 3)
    assert_equal(s, '[0.67]')