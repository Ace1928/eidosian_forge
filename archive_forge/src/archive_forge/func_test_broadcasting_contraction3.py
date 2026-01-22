import numpy as np
import pytest
from opt_einsum import contract, contract_expression
def test_broadcasting_contraction3():
    a = np.random.rand(1, 5, 4)
    b = np.random.rand(4, 1, 6)
    c = np.random.rand(5, 6)
    d = np.random.rand(7, 7)
    ein = contract('ajk,kbl,jl,ab->ab', a, b, c, d, optimize=False)
    opt = contract('ajk,kbl,jl,ab->ab', a, b, c, d, optimize=True)
    assert np.allclose(ein, opt)