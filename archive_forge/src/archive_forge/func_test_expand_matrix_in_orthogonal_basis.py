import itertools
import numpy as np
import pytest
import scipy.linalg
import cirq
@pytest.mark.parametrize('m,basis', itertools.product((I, X, Y, Z, H, SQRT_X, SQRT_Y, SQRT_Z), (PAULI_BASIS, STANDARD_BASIS)))
def test_expand_matrix_in_orthogonal_basis(m, basis):
    expansion = cirq.expand_matrix_in_orthogonal_basis(m, basis)
    reconstructed = np.zeros(m.shape, dtype=complex)
    for name, coefficient in expansion.items():
        reconstructed += coefficient * basis[name]
    assert np.allclose(m, reconstructed)