import itertools
import numpy as np
import pytest
import scipy.linalg
import cirq
@pytest.mark.parametrize('m1,basis', itertools.product((I, X, Y, Z, H, SQRT_X, SQRT_Y, SQRT_Z, E00, E01, E10, E11), (PAULI_BASIS, STANDARD_BASIS)))
def test_expand_is_inverse_of_reconstruct(m1, basis):
    c1 = cirq.expand_matrix_in_orthogonal_basis(m1, basis)
    m2 = cirq.matrix_from_basis_coefficients(c1, basis)
    c2 = cirq.expand_matrix_in_orthogonal_basis(m2, basis)
    assert np.allclose(m1, m2)
    assert c1 == c2