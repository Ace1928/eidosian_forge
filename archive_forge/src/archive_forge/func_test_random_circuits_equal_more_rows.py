import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_random_circuits_equal_more_rows():
    circuit = cirq.testing.random_circuit(qubits=cirq.GridQubit.rect(3, 2), n_moments=6, op_density=1.0)
    qubits = circuit.all_qubits()
    assert_same_output_as_dense(circuit, qubits)