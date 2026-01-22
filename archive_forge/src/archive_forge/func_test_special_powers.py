from mpmath import *
def test_special_powers():
    assert inf ** 3 == inf
    assert isnan(inf ** 0)
    assert inf ** (-3) == 0
    assert (-inf) ** 2 == inf
    assert (-inf) ** 3 == -inf
    assert isnan((-inf) ** 0)
    assert (-inf) ** (-2) == 0
    assert (-inf) ** (-3) == 0
    assert isnan(nan ** 5)
    assert isnan(nan ** 0)