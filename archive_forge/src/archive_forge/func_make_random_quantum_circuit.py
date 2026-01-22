import itertools
from typing import Sequence
import numpy as np
import pytest
import cirq
def make_random_quantum_circuit(qubits: Sequence[cirq.Qid], depth: int) -> cirq.Circuit:
    SQ_GATES = [cirq.X ** 0.5, cirq.Y ** 0.5, cirq.T]
    circuit = cirq.Circuit()
    cz_start = 0
    for q in qubits:
        circuit.append(cirq.H(q))
    for _ in range(depth):
        for q in qubits:
            random_gate = SQ_GATES[np.random.randint(len(SQ_GATES))]
            circuit.append(random_gate(q))
        for q0, q1 in zip(itertools.islice(qubits, cz_start, None, 2), itertools.islice(qubits, cz_start + 1, None, 2)):
            circuit.append(cirq.CNOT(q0, q1))
        cz_start = 1 - cz_start
    for q in qubits:
        circuit.append(cirq.H(q))
    return circuit