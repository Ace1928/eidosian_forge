import itertools
import contextlib
import operator
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_raises, assert_equal
def test_floordiv_128_64():
    with exc_iter(INT128_VALUES, INT64_POS_VALUES) as it:
        for a, b in it:
            c = a // b
            d = mt.extint_floordiv_128_64(a, b)
            if c != d:
                assert_equal(d, c)