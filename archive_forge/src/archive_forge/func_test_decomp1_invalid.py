import itertools
import numpy as np
import pytest
import cirq
import sympy
@pytest.mark.parametrize('u', ZERO_UNITARIES + TWO_SQRT_ISWAP_UNITARIES + THREE_SQRT_ISWAP_UNITARIES)
def test_decomp1_invalid(u):
    q0, q1 = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='cannot be decomposed into exactly 1 sqrt-iSWAP gates'):
        cirq.two_qubit_matrix_to_sqrt_iswap_operations(q0, q1, u, required_sqrt_iswap_count=1)