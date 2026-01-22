import numpy as np
import pytest
from opt_einsum import contract, contract_expression
def test_can_blas_on_healed_broadcast_dimensions():
    expr = contract_expression('ab,bc,bd->acd', (5, 4), (1, 5), (4, 20))
    assert expr.contraction_list[0][2] == 'bc,ab->bca'
    assert expr.contraction_list[0][-1] is False
    assert expr.contraction_list[1][2] == 'bca,bd->acd'
    assert expr.contraction_list[1][-1] == 'GEMM'