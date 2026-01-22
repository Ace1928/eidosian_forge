from thinc.api import (
def test_slanted_triangular_rate():
    rates = slanted_triangular(1.0, 20.0, ratio=10)
    rate0 = next(rates)
    assert rate0 < 1.0
    rate1 = next(rates)
    assert rate1 > rate0
    rate2 = next(rates)
    assert rate2 < rate1
    rate3 = next(rates)
    assert rate0 < rate3 < rate2