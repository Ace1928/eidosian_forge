import numbers
from typing import List
import numpy as np
import pytest
import sympy
import cirq
from cirq.ops.dense_pauli_string import _vectorized_pauli_mul_phase
def test_item_immutable():
    p = -cirq.DensePauliString('XYIZ')
    assert p[-1] == cirq.Z
    assert p[0] == cirq.X
    assert p[1] == cirq.Y
    assert p[2] == cirq.I
    assert p[3] == cirq.Z
    with pytest.raises(TypeError):
        _ = p['test']
    with pytest.raises(IndexError):
        _ = p[4]
    with pytest.raises(TypeError):
        p[2] = cirq.X
    with pytest.raises(TypeError):
        p[:] = p
    assert p[:] == abs(p)
    assert p[1:] == cirq.DensePauliString('YIZ')
    assert p[::2] == cirq.DensePauliString('XI')