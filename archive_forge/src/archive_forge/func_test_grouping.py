import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_grouping():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.X(q0) ** 0.1, cirq.Y(q1) ** 0.2, cirq.Z(q2) ** 0.3, cirq.CNOT(q0, q1), cirq.Y(q1) ** 0.4)
    groupings = [None, {q0: 0, q1: 1, q2: 2}, {q0: 0, q1: 0, q2: 1}, {q0: 0, q1: 1, q2: 0}, {q0: 1, q1: 0, q2: 0}, {q0: 0, q1: 0, q2: 0}]
    for grouping in groupings:
        for initial_state in range(2 * 2 * 2):
            assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1, q2], initial_state=initial_state, grouping=grouping)