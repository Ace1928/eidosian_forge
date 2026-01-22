import numpy as np
import pytest
import cirq
def test_fidelity_invariant_under_unitary_transformation():
    np.testing.assert_allclose(cirq.fidelity(cirq.density_matrix(MAT1), MAT2), cirq.fidelity(cirq.density_matrix(U @ MAT1 @ U.T.conj()), U @ MAT2 @ U.T.conj()))