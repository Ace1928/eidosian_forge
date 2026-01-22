import itertools
import numpy as np
import pytest
import cirq
import sympy
@pytest.mark.parametrize('u', ALL_REGION_UNITARIES)
def test_all_weyl_regions(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqrt_iswap_operations(q0, q1, u, clean_operations=True)
    assert_valid_decomp(u, ops, single_qubit_gate_types=(cirq.PhasedXZGate,))
    ops = cirq.two_qubit_matrix_to_sqrt_iswap_operations(q0, q1, u)
    assert_valid_decomp(u, ops)