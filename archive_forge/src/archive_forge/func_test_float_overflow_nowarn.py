import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_float_overflow_nowarn(self):
    repr(np.array([10000.0, 0.1], dtype='f2'))