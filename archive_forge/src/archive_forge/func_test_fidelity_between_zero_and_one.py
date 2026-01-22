import numpy as np
import pytest
import cirq
def test_fidelity_between_zero_and_one():
    assert 0 <= cirq.fidelity(VEC1, VEC2) <= 1
    assert 0 <= cirq.fidelity(VEC1, MAT1) <= 1
    assert 0 <= cirq.fidelity(cirq.density_matrix(MAT1), MAT2) <= 1