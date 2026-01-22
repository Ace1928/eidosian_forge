import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_unicode_object_array():
    expected = "array(['é'], dtype=object)"
    x = np.array(['é'], dtype=object)
    assert_equal(repr(x), expected)