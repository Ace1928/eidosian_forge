import itertools
import pytest
import sympy
import cirq
def test_to_resolvers_sweep():
    sweep = cirq.Linspace('a', 0, 1, 10)
    assert list(cirq.to_resolvers(sweep)) == list(sweep)