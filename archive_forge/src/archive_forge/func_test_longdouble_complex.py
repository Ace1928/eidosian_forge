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
def test_longdouble_complex():
    x = np.longdouble(1)
    assert x + 1j == 1 + 1j
    assert 1j + x == 1 + 1j