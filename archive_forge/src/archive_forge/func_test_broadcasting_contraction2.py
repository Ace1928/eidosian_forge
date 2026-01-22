import numpy as np
import pytest
from opt_einsum import contract, contract_expression
def test_broadcasting_contraction2():
    a = np.random.rand(1, 1, 5, 4)
    b = np.random.rand(4, 6)
    c = np.random.rand(5, 6)
    d = np.random.rand(7, 7)
    ein_scalar = contract('abjk,kl,jl', a, b, c, optimize=False)
    opt_scalar = contract('abjk,kl,jl', a, b, c, optimize=True)
    assert np.allclose(ein_scalar, opt_scalar)
    result = ein_scalar * d
    ein = contract('abjk,kl,jl,ab->ab', a, b, c, d, optimize=False)
    opt = contract('abjk,kl,jl,ab->ab', a, b, c, d, optimize=True)
    assert np.allclose(ein, result)
    assert np.allclose(opt, result)