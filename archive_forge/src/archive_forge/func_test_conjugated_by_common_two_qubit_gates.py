import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_conjugated_by_common_two_qubit_gates():

    class OrderSensitiveGate(cirq.Gate):

        def num_qubits(self):
            return 2

        def _decompose_(self, qubits):
            return [cirq.Y(qubits[0]) ** (-0.5), cirq.CNOT(*qubits)]
    a, b, c, d = cirq.LineQubit.range(4)
    two_qubit_gates = [cirq.CNOT, cirq.CZ, cirq.ISWAP, cirq.ISWAP_INV, cirq.SWAP, cirq.XX ** 0.5, cirq.YY ** 0.5, cirq.ZZ ** 0.5, cirq.XX, cirq.YY, cirq.ZZ, cirq.XX ** (-0.5), cirq.YY ** (-0.5), cirq.ZZ ** (-0.5)]
    two_qubit_gates.extend([OrderSensitiveGate()])
    for p1 in [cirq.I, cirq.X, cirq.Y, cirq.Z]:
        for p2 in [cirq.I, cirq.X, cirq.Y, cirq.Z]:
            pd = cirq.DensePauliString([p1, p2])
            p = pd.sparse()
            for g in two_qubit_gates:
                assert p.conjugated_by(g.on(c, d)) == p
                actual = cirq.unitary(p.conjugated_by(g.on(a, b)).dense([a, b]))
                u = cirq.unitary(g)
                expected = np.conj(u.T) @ cirq.unitary(pd) @ u
                np.testing.assert_allclose(actual, expected, atol=1e-08)