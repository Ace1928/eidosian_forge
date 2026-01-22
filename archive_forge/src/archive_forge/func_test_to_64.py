import itertools
import contextlib
import operator
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_raises, assert_equal
def test_to_64():
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            if not INT64_MIN <= a <= INT64_MAX:
                assert_raises(OverflowError, mt.extint_to_64, a)
            else:
                b = mt.extint_to_64(a)
                if a != b:
                    assert_equal(b, a)