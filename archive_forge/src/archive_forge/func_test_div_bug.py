from mpmath import *
def test_div_bug():
    assert isnan(nan / 1)
    assert isnan(nan / 2)
    assert inf / 2 == inf
    assert -inf / 2 == -inf