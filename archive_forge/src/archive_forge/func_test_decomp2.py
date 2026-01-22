import itertools
import numpy as np
import pytest
import cirq
import sympy
@pytest.mark.parametrize('u', ZERO_UNITARIES + ONE_SQRT_ISWAP_UNITARIES + TWO_SQRT_ISWAP_UNITARIES)
def test_decomp2(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqrt_iswap_operations(q0, q1, u, required_sqrt_iswap_count=2)
    assert_valid_decomp(u, ops)
    assert_specific_sqrt_iswap_count(ops, 2)