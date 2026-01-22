import itertools
import contextlib
import operator
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_raises, assert_equal
@pytest.mark.slow
def test_divmod_128_64():
    with exc_iter(INT128_VALUES, INT64_POS_VALUES) as it:
        for a, b in it:
            if a >= 0:
                c, cr = divmod(a, b)
            else:
                c, cr = divmod(-a, b)
                c = -c
                cr = -cr
            d, dr = mt.extint_divmod_128_64(a, b)
            if c != d or d != dr or b * d + dr != a:
                assert_equal(d, c)
                assert_equal(dr, cr)
                assert_equal(b * d + dr, a)