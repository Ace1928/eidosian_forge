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
def test_equal_nbytes(self):
    for type in types:
        x = type(0)
        assert_(sys.getsizeof(x) > x.nbytes)