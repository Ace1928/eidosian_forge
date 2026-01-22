import pytest
import sympy
import cirq
def test_linspace_sympy_symbol():
    a = sympy.Symbol('a')
    sweep = cirq.Linspace(a, 0.34, 9.16, 7)
    assert len(sweep) == 7
    params = list(sweep.param_tuples())
    assert len(params) == 7
    assert params[0] == (('a', 0.34),)
    assert params[-1] == (('a', 9.16),)