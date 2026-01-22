from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXGate, RYGate, RZGate

    Given a pair of distinct indices 0 ≤ (first_index, second_index) ≤ 2,
    produces a two-qubit circuit which rotates a canonical gate

        a0 XX + a1 YY + a2 ZZ

    into

        a[first] XX + a[second] YY + a[other] ZZ .
    