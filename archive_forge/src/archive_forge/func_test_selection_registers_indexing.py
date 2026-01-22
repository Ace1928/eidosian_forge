import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.infra import split_qubits, merge_qubits, get_named_qubits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('n, N, m, M', [(4, 10, 5, 19), (4, 16, 5, 32)])
@allow_deprecated_cirq_ft_use_in_tests
def test_selection_registers_indexing(n, N, m, M):
    regs = [cirq_ft.SelectionRegister('x', n, N), cirq_ft.SelectionRegister('y', m, M)]
    for x in range(regs[0].iteration_length):
        for y in range(regs[1].iteration_length):
            assert np.ravel_multi_index((x, y), (N, M)) == x * M + y
            assert np.unravel_index(x * M + y, (N, M)) == (x, y)
    assert np.prod(tuple((reg.iteration_length for reg in regs))) == N * M