import numbers
from typing import List
import numpy as np
import pytest
import sympy
import cirq
from cirq.ops.dense_pauli_string import _vectorized_pauli_mul_phase
def test_idiv():
    p = cirq.MutableDensePauliString('XYZ', coefficient=2)
    p /= 2
    assert p == cirq.MutableDensePauliString('XYZ')
    with pytest.raises(TypeError):
        p /= object()