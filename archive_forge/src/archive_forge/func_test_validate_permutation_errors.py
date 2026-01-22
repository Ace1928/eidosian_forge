import random
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_validate_permutation_errors():
    validate_permutation = cca.PermutationGate.validate_permutation
    validate_permutation({})
    with pytest.raises(IndexError, match='key and value sets must be the same\\.'):
        validate_permutation({0: 2, 1: 3})
    with pytest.raises(IndexError, match='keys of the permutation must be non-negative\\.'):
        validate_permutation({-1: 0, 0: -1})
    with pytest.raises(IndexError, match='key is out of bounds\\.'):
        validate_permutation({0: 3, 3: 0}, 2)
    gate = cca.SwapPermutationGate()
    assert cirq.circuit_diagram_info(gate, default=None) is None