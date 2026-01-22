import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_fsim_resolve(resolve_fn):
    f = cirq.FSimGate(sympy.Symbol('a'), sympy.Symbol('b'))
    assert cirq.is_parameterized(f)
    f = resolve_fn(f, {'a': 2})
    assert f == cirq.FSimGate(2, sympy.Symbol('b'))
    assert cirq.is_parameterized(f)
    f = resolve_fn(f, {'b': 1})
    assert f == cirq.FSimGate(2, 1)
    assert not cirq.is_parameterized(f)