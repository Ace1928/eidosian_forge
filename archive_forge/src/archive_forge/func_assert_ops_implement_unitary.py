import cmath
import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.analytical_decompositions.two_qubit_to_cz import (
from cirq.testing import random_two_qubit_circuit_with_czs
def assert_ops_implement_unitary(q0, q1, operations, intended_effect, atol=0.01):
    actual_effect = _operations_to_matrix(operations, (q0, q1))
    assert cirq.allclose_up_to_global_phase(actual_effect, intended_effect, atol=atol)