import pytest
import cirq
import numpy as np
from cirq.ops import QubitPermutationGate
def test_permutation_gate_repr():
    cirq.testing.assert_equivalent_repr(QubitPermutationGate([0, 1]))