import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_num_two_qubit_gates_required():
    for i in range(4):
        assert cirq.num_cnots_required(cirq.testing.random_two_qubit_circuit_with_czs(i).unitary(), atol=1e-06) == i
    assert cirq.num_cnots_required(np.eye(4)) == 0