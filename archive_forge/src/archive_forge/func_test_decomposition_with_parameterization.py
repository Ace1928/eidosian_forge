from typing import List
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_decomposition_with_parameterization(n):
    angles = sympy.symbols([f'x_{i}' for i in range(2 ** n)])
    exponent = sympy.Symbol('e')
    diagonal_gate = cirq.DiagonalGate(angles) ** exponent
    parameterized_op = diagonal_gate(*cirq.LineQubit.range(n))
    decomposed_circuit = cirq.Circuit(cirq.decompose(parameterized_op))
    for exponent_value in [-0.5, 0.5, 1]:
        for i in range(len(_candidate_angles) - 2 ** n + 1):
            resolver = {exponent: exponent_value}
            resolver.update({angles[j]: x_j for j, x_j in enumerate(_candidate_angles[i:i + 2 ** n])})
            resolved_op = cirq.resolve_parameters(parameterized_op, resolver)
            resolved_circuit = cirq.resolve_parameters(decomposed_circuit, resolver)
            np.testing.assert_allclose(cirq.unitary(resolved_op), cirq.unitary(resolved_circuit), atol=1e-08)