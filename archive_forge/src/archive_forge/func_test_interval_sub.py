from sympy.plotting.intervalmath import interval
from sympy.testing.pytest import raises
def test_interval_sub():
    assert (interval(1, 2) - interval(1, 5) == interval(-4, 1)) == (True, True)
    assert (interval(1, 2) - 1 == interval(0, 1)) == (True, True)
    assert (1 - interval(1, 2) == interval(-1, 0)) == (True, True)
    a = 1 - interval(1, 2, is_valid=False)
    assert a.is_valid is False
    a = interval(1, 4, is_valid=None) - 1
    assert a.is_valid is None
    a = interval(1, 3, is_valid=False) - interval(1, 3)
    assert a.is_valid is False
    a = interval(1, 3, is_valid=None) - interval(1, 3)
    assert a.is_valid is None