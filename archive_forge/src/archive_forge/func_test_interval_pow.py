from sympy.external import import_module
from sympy.plotting.intervalmath import (
def test_interval_pow():
    a = 2 ** interval(1, 2) == interval(2, 4)
    assert a == (True, True)
    a = interval(1, 2) ** interval(1, 2) == interval(1, 4)
    assert a == (True, True)
    a = interval(-1, 1) ** interval(0.5, 2)
    assert a.is_valid is None
    a = interval(-2, -1) ** interval(1, 2)
    assert a.is_valid is False
    a = interval(-2, -1) ** (1.0 / 2)
    assert a.is_valid is False
    a = interval(-1, 1) ** (1.0 / 2)
    assert a.is_valid is None
    a = interval(-1, 1) ** (1.0 / 3) == interval(-1, 1)
    assert a == (True, True)
    a = interval(-1, 1) ** 2 == interval(0, 1)
    assert a == (True, True)
    a = interval(-1, 1) ** (1.0 / 29) == interval(-1, 1)
    assert a == (True, True)
    a = -2 ** interval(1, 1) == interval(-2, -2)
    assert a == (True, True)
    a = interval(1, 2, is_valid=False) ** 2
    assert a.is_valid is False
    a = (-3) ** interval(1, 2)
    assert a.is_valid is False
    a = (-4) ** interval(0.5, 0.5)
    assert a.is_valid is False
    assert ((-3) ** interval(1, 1) == interval(-3, -3)) == (True, True)
    a = interval(8, 64) ** (2.0 / 3)
    assert abs(a.start - 4) < 1e-10
    assert abs(a.end - 16) < 1e-10
    a = interval(-8, 64) ** (2.0 / 3)
    assert abs(a.start - 4) < 1e-10
    assert abs(a.end - 16) < 1e-10