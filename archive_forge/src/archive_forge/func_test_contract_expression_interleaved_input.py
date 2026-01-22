import numpy as np
import pytest
from opt_einsum import contract, contract_expression, contract_path, helpers
from opt_einsum.paths import linear_to_ssa, ssa_to_linear
def test_contract_expression_interleaved_input():
    x, y, z = (np.random.randn(2, 2) for _ in 'xyz')
    expected = np.einsum(x, [0, 1], y, [1, 2], z, [2, 3], [3, 0])
    xshp, yshp, zshp = ((2, 2) for _ in 'xyz')
    expr = contract_expression(xshp, [0, 1], yshp, [1, 2], zshp, [2, 3], [3, 0])
    out = expr(x, y, z)
    assert np.allclose(out, expected)