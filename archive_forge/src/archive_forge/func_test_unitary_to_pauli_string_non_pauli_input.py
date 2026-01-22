from typing import cast
import itertools
import cmath
import pytest
import numpy as np
from cirq.ops import DensePauliString, T
from cirq import protocols
from cirq.transformers.analytical_decompositions import unitary_to_pauli_string
def test_unitary_to_pauli_string_non_pauli_input():
    got = unitary_to_pauli_string(protocols.unitary(T))
    assert got is None
    got = unitary_to_pauli_string(np.array([[1, 0], [1, 0]]))
    assert got is None
    got = unitary_to_pauli_string(np.array([[1, 1], [0, 2]]))
    assert got is None
    got = unitary_to_pauli_string(np.array([[0, 0.5], [1, -1]]), eps=1.1)
    assert got is None