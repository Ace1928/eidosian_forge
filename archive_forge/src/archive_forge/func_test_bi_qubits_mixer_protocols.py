import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('adjoint', [False, True])
@allow_deprecated_cirq_ft_use_in_tests
def test_bi_qubits_mixer_protocols(adjoint: bool):
    g = cirq_ft.algos.BiQubitsMixer(adjoint=adjoint)
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(g)
    cirq.testing.assert_equivalent_repr(g, setup_code='import cirq_ft')
    assert g ** 1 is g
    assert g ** (-1) == cirq_ft.algos.BiQubitsMixer(adjoint=adjoint ^ True)