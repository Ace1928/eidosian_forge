import itertools
import numpy as np
import pytest
import scipy.linalg
import cirq
@pytest.mark.parametrize('basis1,basis2', ((PAULI_BASIS, cirq.kron_bases(PAULI_BASIS)), (STANDARD_BASIS, cirq.kron_bases(STANDARD_BASIS, repeat=1)), (cirq.kron_bases(PAULI_BASIS, PAULI_BASIS), cirq.kron_bases(PAULI_BASIS, repeat=2)), (cirq.kron_bases(cirq.kron_bases(PAULI_BASIS, repeat=2), cirq.kron_bases(PAULI_BASIS, repeat=3), PAULI_BASIS), cirq.kron_bases(PAULI_BASIS, repeat=6)), (cirq.kron_bases(cirq.kron_bases(PAULI_BASIS, STANDARD_BASIS), cirq.kron_bases(PAULI_BASIS, STANDARD_BASIS)), cirq.kron_bases(PAULI_BASIS, STANDARD_BASIS, repeat=2))))
def test_kron_bases_consistency(basis1, basis2):
    assert set(basis1.keys()) == set(basis2.keys())
    for name in basis1.keys():
        assert np.all(basis1[name] == basis2[name])