import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('qubits', [cirq.LineQubit.range(2), cirq.LineQubit.range(4)])
@pytest.mark.parametrize('split', [True, False])
def test_clifford_circuit_2(qubits, split):
    circuit = cirq.Circuit()
    np.random.seed(2)
    for _ in range(50):
        x = np.random.randint(7)
        if x == 0:
            circuit.append(cirq.X(np.random.choice(qubits)))
        elif x == 1:
            circuit.append(cirq.Z(np.random.choice(qubits)))
        elif x == 2:
            circuit.append(cirq.Y(np.random.choice(qubits)))
        elif x == 3:
            circuit.append(cirq.S(np.random.choice(qubits)))
        elif x == 4:
            circuit.append(cirq.H(np.random.choice(qubits)))
        elif x == 5:
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        elif x == 6:
            circuit.append(cirq.CZ(qubits[0], qubits[1]))
    circuit.append(cirq.measure(qubits[0]))
    result = cirq.CliffordSimulator(split_untangled_states=split).run(circuit, repetitions=100)
    assert sum(result.measurements['q(0)'])[0] < 80
    assert sum(result.measurements['q(0)'])[0] > 20