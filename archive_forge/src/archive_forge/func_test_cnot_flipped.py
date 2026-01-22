import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_cnot_flipped():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(q1, q0))
    for initial_state in range(4):
        assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1], initial_state=initial_state)