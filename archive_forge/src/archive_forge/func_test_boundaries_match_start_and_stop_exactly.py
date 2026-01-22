import pytest
from numpy import (
from numpy.testing import (
def test_boundaries_match_start_and_stop_exactly(self):
    start = 0.3
    stop = 20.3
    y = geomspace(start, stop, num=1)
    assert_equal(y[0], start)
    y = geomspace(start, stop, num=1, endpoint=False)
    assert_equal(y[0], start)
    y = geomspace(start, stop, num=3)
    assert_equal(y[0], start)
    assert_equal(y[-1], stop)
    y = geomspace(start, stop, num=3, endpoint=False)
    assert_equal(y[0], start)