import numpy as np
import pytest
import sympy
import cirq
def test_parameterized_decompose():
    angles = sympy.symbols('x0, x1, x2, x3')
    parameterized_op = cirq.TwoQubitDiagonalGate(angles).on(*cirq.LineQubit.range(2))
    decomposed_circuit = cirq.Circuit(cirq.decompose(parameterized_op))
    for resolver in cirq.Linspace('x0', -2, 2, 3) * cirq.Linspace('x1', -2, 2, 3) * cirq.Linspace('x2', -2, 2, 3) * cirq.Linspace('x3', -2, 2, 3):
        np.testing.assert_allclose(cirq.unitary(cirq.resolve_parameters(parameterized_op, resolver)), cirq.unitary(cirq.resolve_parameters(decomposed_circuit, resolver)))