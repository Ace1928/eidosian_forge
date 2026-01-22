import cmath
import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.analytical_decompositions.two_qubit_to_cz import (
from cirq.testing import random_two_qubit_circuit_with_czs
def test_kak_decomposition_depth_partial_cz():
    a, b = cirq.LineQubit.range(2)
    u = cirq.testing.random_unitary(4)
    operations_with_full = cirq.two_qubit_matrix_to_cz_operations(a, b, u, True)
    c = cirq.Circuit(operations_with_full)
    assert len(c) <= 8
    u = cirq.unitary(cirq.Circuit(cirq.CNOT(a, b), cirq.CNOT(b, a)))
    operations_with_part = cirq.two_qubit_matrix_to_cz_operations(a, b, u, True)
    c = cirq.Circuit(operations_with_part)
    assert len(c) <= 6
    u = cirq.unitary(cirq.CNOT ** 0.1)
    operations_with_part = cirq.two_qubit_matrix_to_cz_operations(a, b, u, True)
    c = cirq.Circuit(operations_with_part)
    assert len(c) <= 4
    u = cirq.unitary(cirq.ControlledGate(cirq.Y))
    operations_with_part = cirq.two_qubit_matrix_to_cz_operations(a, b, u, True)
    c = cirq.Circuit(operations_with_part)
    assert len(c) <= 4