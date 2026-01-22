import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_with_qubits():
    old_qubits = cirq.LineQubit.range(9)
    new_qubits = cirq.LineQubit.range(9, 18)
    qubit_pauli_map = {q: cirq.Pauli.by_index(q.x) for q in old_qubits}
    pauli_string = cirq.PauliString(qubit_pauli_map, -1)
    new_pauli_string = pauli_string.with_qubits(*new_qubits)
    assert new_pauli_string.qubits == tuple(new_qubits)
    for q in new_qubits:
        assert new_pauli_string[q] == cirq.Pauli.by_index(q.x)
    assert new_pauli_string.coefficient == -1