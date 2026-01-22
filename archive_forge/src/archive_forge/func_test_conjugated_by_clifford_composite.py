import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_conjugated_by_clifford_composite():

    class UnknownGate(cirq.Gate):

        def num_qubits(self) -> int:
            return 4

        def _decompose_(self, qubits):
            yield cirq.SWAP(qubits[0], qubits[1])
            yield cirq.SWAP(qubits[2], qubits[3])
    a, b, c, d = cirq.LineQubit.range(4)
    p = cirq.X(a) * cirq.Z(b)
    u = UnknownGate()
    assert p.conjugated_by(u(a, b, c, d)) == cirq.Z(a) * cirq.X(b)