import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('dim', [*range(2, 10)])
@allow_deprecated_cirq_ft_use_in_tests
def test_prepare_t_complexity(dim):
    prepare = cirq_ft.PrepareHubbard(x_dim=dim, y_dim=dim, t=2, mu=8)
    cost = cirq_ft.t_complexity(prepare)
    logN = 2 * (dim - 1).bit_length() + 1
    assert cost.t <= 32 * logN
    assert cost.rotations <= 2 * logN + 9