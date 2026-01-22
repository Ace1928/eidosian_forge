import itertools
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('resolve_fn, global_shift', [(cirq.resolve_parameters, 0), (cirq.resolve_parameters_once, 0.1)])
def test_parameterize(resolve_fn, global_shift):
    parameterized_gate = cirq.PhasedXPowGate(exponent=sympy.Symbol('a'), phase_exponent=sympy.Symbol('b'), global_shift=global_shift)
    assert cirq.pow(parameterized_gate, 5) == cirq.PhasedXPowGate(exponent=sympy.Symbol('a') * 5, phase_exponent=sympy.Symbol('b'), global_shift=global_shift)
    assert cirq.unitary(parameterized_gate, default=None) is None
    assert cirq.is_parameterized(parameterized_gate)
    q = cirq.NamedQubit('q')
    parameterized_decomposed_circuit = cirq.Circuit(cirq.decompose(parameterized_gate(q)))
    for resolver in cirq.Linspace('a', 0, 2, 10) * cirq.Linspace('b', 0, 2, 10):
        resolved_gate = resolve_fn(parameterized_gate, resolver)
        assert resolved_gate == cirq.PhasedXPowGate(exponent=resolver.value_of('a'), phase_exponent=resolver.value_of('b'), global_shift=global_shift)
        np.testing.assert_allclose(cirq.unitary(resolved_gate(q)), cirq.unitary(resolve_fn(parameterized_decomposed_circuit, resolver)), atol=1e-08)
    unparameterized_gate = cirq.PhasedXPowGate(exponent=0.1, phase_exponent=0.2, global_shift=global_shift)
    assert not cirq.is_parameterized(unparameterized_gate)
    assert cirq.is_parameterized(unparameterized_gate ** sympy.Symbol('a'))
    assert cirq.is_parameterized(unparameterized_gate ** (sympy.Symbol('a') + 1))
    resolver = {'a': 0.5j}
    with pytest.raises(ValueError, match='complex value'):
        resolve_fn(cirq.PhasedXPowGate(exponent=sympy.Symbol('a'), phase_exponent=0.2, global_shift=global_shift), resolver)
    with pytest.raises(ValueError, match='complex value'):
        resolve_fn(cirq.PhasedXPowGate(exponent=0.1, phase_exponent=sympy.Symbol('a'), global_shift=global_shift), resolver)