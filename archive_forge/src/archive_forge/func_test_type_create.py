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
def test_type_create(self):
    for k, atype in enumerate(types):
        a = np.array([1, 2, 3], atype)
        b = atype([1, 2, 3])
        assert_equal(a, b)