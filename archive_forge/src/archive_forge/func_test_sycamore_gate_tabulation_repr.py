import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.heuristic_decompositions.two_qubit_gate_tabulation import (
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
from cirq.testing import random_special_unitary, assert_equivalent_repr
def test_sycamore_gate_tabulation_repr():
    simple_tabulation = TwoQubitGateTabulation(np.array([[1 + 0j, 0j, 0j, 0j]], dtype=np.complex128), np.array([[1 + 0j, 0j, 0j, 0j]], dtype=np.complex128), [[]], 0.49, 'Sample string', ())
    assert_equivalent_repr(simple_tabulation)