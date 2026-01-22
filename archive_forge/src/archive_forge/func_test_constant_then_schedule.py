from thinc.api import (
def test_constant_then_schedule():
    rates = constant_then(1.0, 2, [100, 200])
    assert next(rates) == 1.0
    assert next(rates) == 1.0
    assert next(rates) == 100
    assert next(rates) == 200