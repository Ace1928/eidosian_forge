from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXGate, RYGate, RZGate
def canonical_rotation_circuit(first_index, second_index):
    """
    Given a pair of distinct indices 0 ≤ (first_index, second_index) ≤ 2,
    produces a two-qubit circuit which rotates a canonical gate

        a0 XX + a1 YY + a2 ZZ

    into

        a[first] XX + a[second] YY + a[other] ZZ .
    """
    conj = QuantumCircuit(2)
    if (0, 1) == (first_index, second_index):
        pass
    elif (0, 2) == (first_index, second_index):
        conj.rx(-np.pi / 2, [0])
        conj.rx(np.pi / 2, [1])
    elif (1, 0) == (first_index, second_index):
        conj.rz(-np.pi / 2, [0])
        conj.rz(-np.pi / 2, [1])
    elif (1, 2) == (first_index, second_index):
        conj.rz(np.pi / 2, [0])
        conj.rz(np.pi / 2, [1])
        conj.ry(np.pi / 2, [0])
        conj.ry(-np.pi / 2, [1])
    elif (2, 0) == (first_index, second_index):
        conj.rz(np.pi / 2, [0])
        conj.rz(np.pi / 2, [1])
        conj.rx(np.pi / 2, [0])
        conj.rx(-np.pi / 2, [1])
    elif (2, 1) == (first_index, second_index):
        conj.ry(np.pi / 2, [0])
        conj.ry(-np.pi / 2, [1])
    return conj