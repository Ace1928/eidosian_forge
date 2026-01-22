import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('adjoint', [False, True])
@allow_deprecated_cirq_ft_use_in_tests
def test_single_qubit_compare_protocols(adjoint: bool):
    g = cirq_ft.algos.SingleQubitCompare(adjoint=adjoint)
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(g)
    cirq.testing.assert_equivalent_repr(g, setup_code='import cirq_ft')
    expected_side = cirq_ft.infra.Side.LEFT if adjoint else cirq_ft.infra.Side.RIGHT
    assert g.signature[2] == cirq_ft.Register('less_than', 1, side=expected_side)
    assert g.signature[3] == cirq_ft.Register('greater_than', 1, side=expected_side)
    with pytest.raises(ValueError):
        _ = g ** 0.5
    assert g ** 2 == cirq.IdentityGate(4)
    assert g ** 1 is g
    assert g ** (-1) == cirq_ft.algos.SingleQubitCompare(adjoint=adjoint ^ True)