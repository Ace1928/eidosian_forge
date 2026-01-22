import itertools
import contextlib
import operator
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_raises, assert_equal
def test_mul_64_64():
    with exc_iter(INT64_VALUES, INT64_VALUES) as it:
        for a, b in it:
            c = a * b
            d = mt.extint_mul_64_64(a, b)
            if c != d:
                assert_equal(d, c)