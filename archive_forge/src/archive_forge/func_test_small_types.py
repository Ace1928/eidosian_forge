import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
def test_small_types(self):
    for t in [np.int8, np.int16, np.float16]:
        a = t(3)
        b = a ** 4
        assert_(b == 81, 'error with %r: got %r' % (t, b))