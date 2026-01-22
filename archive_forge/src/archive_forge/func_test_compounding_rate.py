from thinc.api import (
def test_compounding_rate():
    rates = compounding(1, 16, 1.01)
    rate0 = next(rates)
    assert rate0 == 1.0
    rate1 = next(rates)
    rate2 = next(rates)
    rate3 = next(rates)
    assert rate3 > rate2 > rate1 > rate0
    assert rate3 - rate2 > rate2 - rate1 > rate1 - rate0