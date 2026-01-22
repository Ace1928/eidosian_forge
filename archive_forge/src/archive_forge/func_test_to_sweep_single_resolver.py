import itertools
import pytest
import sympy
import cirq
@pytest.mark.parametrize('r_gen', [lambda: {'a': 1}, lambda: {sympy.Symbol('a'): 1}, lambda: cirq.ParamResolver({'a': 1}), lambda: cirq.ParamResolver({sympy.Symbol('a'): 1})])
def test_to_sweep_single_resolver(r_gen):
    sweep = cirq.to_sweep(r_gen())
    assert isinstance(sweep, cirq.Sweep)
    assert list(sweep) == [cirq.ParamResolver({'a': 1})]