import pytest
from numpy import (
from numpy.testing import (
def test_retstep(self):
    for num in [0, 1, 2]:
        for ept in [False, True]:
            y = linspace(0, 1, num, endpoint=ept, retstep=True)
            assert isinstance(y, tuple) and len(y) == 2
            if num == 2:
                y0_expect = [0.0, 1.0] if ept else [0.0, 0.5]
                assert_array_equal(y[0], y0_expect)
                assert_equal(y[1], y0_expect[1])
            elif num == 1 and (not ept):
                assert_array_equal(y[0], [0.0])
                assert_equal(y[1], 1.0)
            else:
                assert_array_equal(y[0], [0.0][:num])
                assert isnan(y[1])