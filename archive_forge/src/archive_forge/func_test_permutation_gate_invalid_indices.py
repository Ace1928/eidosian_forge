import pytest
import cirq
import numpy as np
from cirq.ops import QubitPermutationGate
def test_permutation_gate_invalid_indices():
    with pytest.raises(ValueError, match='Invalid indices'):
        QubitPermutationGate([1, 0, 2, 4])
    with pytest.raises(ValueError, match='Invalid indices'):
        QubitPermutationGate([-1])