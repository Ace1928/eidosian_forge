from thinc.api import (
def test_cyclic_triangular():
    rates = cyclic_triangular(0.1, 1.0, 2)
    expected = [0.55, 1.0, 0.55, 0.1, 0.55, 1.0, 0.55, 0.1, 0.55, 1.0]
    for i in range(10):
        assert next(rates) == expected[i]