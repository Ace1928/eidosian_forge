import numpy as np
import pytest, sympy
import cirq
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_phase_gradient_symbolic(resolve_fn):
    a = cirq.PhaseGradientGate(num_qubits=2, exponent=0.5)
    b = cirq.PhaseGradientGate(num_qubits=2, exponent=sympy.Symbol('t'))
    assert not cirq.is_parameterized(a)
    assert cirq.is_parameterized(b)
    assert cirq.has_unitary(a)
    assert not cirq.has_unitary(b)
    assert resolve_fn(a, {'t': 0.25}) is a
    assert resolve_fn(b, {'t': 0.5}) == a
    assert resolve_fn(b, {'t': 0.25}) == cirq.PhaseGradientGate(num_qubits=2, exponent=0.25)