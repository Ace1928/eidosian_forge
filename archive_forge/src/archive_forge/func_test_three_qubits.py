import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_three_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.CCX(q0, q1, q2))
    with pytest.raises(ValueError, match='Can only handle 1 and 2 qubit operations'):
        assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1, q2])