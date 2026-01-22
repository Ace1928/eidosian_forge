import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('n', [*range(1, 10)])
@allow_deprecated_cirq_ft_use_in_tests
def test_contiguous_register_gate_t_complexity(n):
    gate = cirq_ft.ContiguousRegisterGate(n, 2 * n)
    toffoli_complexity = cirq_ft.t_complexity(cirq.CCNOT)
    assert cirq_ft.t_complexity(gate) == (n ** 2 + n - 1) * toffoli_complexity