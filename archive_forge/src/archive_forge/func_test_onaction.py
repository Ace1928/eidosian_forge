from sympy.strategies.branch.core import (
def test_onaction():
    L = []

    def record(fn, input, output):
        L.append((input, output))
    list(onaction(inc, record)(2))
    assert L == [(2, 3)]
    list(onaction(identity, record)(2))
    assert L == [(2, 3)]