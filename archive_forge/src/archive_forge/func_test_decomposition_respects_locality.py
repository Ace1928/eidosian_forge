import itertools
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('gate', [cirq.CCX, cirq.CSWAP, cirq.CCZ, cirq.ThreeQubitDiagonalGate([2, 3, 5, 7, 11, 13, 17, 19])])
def test_decomposition_respects_locality(gate):
    a = cirq.GridQubit(0, 0)
    b = cirq.GridQubit(1, 0)
    c = cirq.GridQubit(0, 1)
    dev = cirq.testing.ValidatingTestDevice(qubits={a, b, c}, validate_locality=True)
    for x, y, z in itertools.permutations([a, b, c]):
        circuit = cirq.Circuit(gate(x, y, z))
        circuit = cirq.Circuit(cirq.decompose(circuit))
        dev.validate_circuit(circuit)