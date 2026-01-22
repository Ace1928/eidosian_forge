import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('data', [[np.arange(6, dtype=int).reshape(2, 3), 4 * np.arange(6, dtype=int).reshape(2, 3)], [np.arange(8, dtype=int).reshape(2, 2, 2)]])
@pytest.mark.parametrize('num_controls', [0, 1, 2])
@allow_deprecated_cirq_ft_use_in_tests
def test_ndim_t_complexity(data, num_controls):
    selection_bitsizes = tuple(((s - 1).bit_length() for s in data[0].shape))
    target_bitsizes = tuple((int(np.max(d)).bit_length() for d in data))
    qrom = cirq_ft.QROM(data, selection_bitsizes, target_bitsizes, num_controls=num_controls)
    g = cirq_ft.testing.GateHelper(qrom)
    n = data[0].size
    assert cirq_ft.t_complexity(g.gate) == cirq_ft.t_complexity(g.operation)
    assert cirq_ft.t_complexity(g.gate).t == max(0, 4 * n - 8 + 4 * num_controls)